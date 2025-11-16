import time
from src.langgraphagenticai.state.state import NLUOutput, CallState, TranscriptEntry
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
import pyttsx3
import speech_recognition as sr
from groq import Groq
import tempfile
import os
from typing import List, Dict
ALLOWED_INTENTS: List[str] = [
    "Billing Issue",
    "SIM Not Working",
    "No Network Coverage",
    "Internet Speed Slow",
    "Data Not Working After Recharge",
    "Call Drops Frequently",
]
class CallCenterNode:
    def __init__(self, model=None, llm=None):
        # Ensure self.llm is a ChatGroq instance with invoke()/with_structured_output()
        self.llm = llm or self.get_llm_model(model)

    def get_llm_model(self, model=None):
        # Return the actual ChatGroq instance
        return GroqLLM(model=model or "openai/gpt-oss-20b").get_llm_model()



    def preprocess_node(self,state: CallState) -> CallState:
        """Cleans the latest transcript entry and updates clean_text."""
        if state['transcript']:
            state['clean_text'] = state['transcript'][-1]['text'].strip().lower()
        else:
            state['clean_text'] = "" # Handle case of empty transcript
            
        # Initialize NLU fields for safety
        state['intent'] = "Unknown"
        state['confidence'] = 0.0
        state['entities'] = {}
        
        return state

    def nlu_node(self, state: CallState) -> CallState:
        intents = ALLOWED_INTENTS
        prompt = f"""
You are an NLU module for a telecom call center.

Task:
1) Classify the user's intent into exactly ONE of these intents (must pick one): {intents}
2) Extract entities as key:value pairs (e.g., account_number, recharge_amount, date(YYYY-MM-DD), location, device_model, error_code)
3) Return ONLY a single JSON object matching this schema:
{{
  "intent": "<one of: {intents}>",
  "confidence": <float 0.0..1.0>,
  "entities": {{ "<key>": "<value>" }},
  "notes": "<optional short note>"
}}

Rules:
- Output must be valid JSON only (no markdown, no extra text).
- Always choose the best matching intent from the list.
- Normalize numbers by removing non-digits; use ISO date when possible.

User Input: "{state.get('clean_text','')}"
"""
        structured_llm = self.llm.with_structured_output(NLUOutput)

        def fallback_match(txt: str) -> str:
            txt = (txt or "").lower()
            best_intent, best_score = None, -1
            for intent, kws in INTENT_KEYWORDS.items():
                score = sum(1 for kw in kws if kw in txt)
                if score > best_score:
                    best_score, best_intent = score, intent
            return best_intent or "Billing Issue"

        try:
            nlu_result: NLUOutput = structured_llm.invoke(prompt)
            intent = str(nlu_result.intent)
            # Safety: enforce membership
            if intent not in ALLOWED_INTENTS:
                intent = fallback_match(state.get('clean_text', ''))
            state['intent'] = intent

            # Clamp confidence 0..1, ensure min > 0 to avoid 0% displays
            try:
                conf = float(getattr(nlu_result, "confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.01, min(conf, 1.0))
            state['confidence'] = conf

            ents = getattr(nlu_result, "entities", {}) or {}
            state['entities'] = dict(ents)
        except Exception as e:
            # Hard fallback → deterministic keyword routing
            intent = fallback_match(state.get('clean_text', ''))
            state['intent'] = intent
            state['confidence'] = 0.5  # conservative default
            state['entities'] = {}

        return state

    def billing_issue_node(self, state: CallState) -> CallState:
        prompt = f"""
    You are a senior telecom billing agent. Read the user input and extracted entities.
    Goal: produce 3 things in plain text separated by newlines (no questions):
    1) A single, one-sentence customer-facing acknowledgement + decisive resolution or next step (what we will do or what the customer should do).
    2) A single short internal action label (choose one): "adjust-bill", "open-billing-ticket", "escalate-to-billing", "inform-no-issue-found", "request-docs" (but do NOT ask the user for docs).
    3) A one-line internal note for logs (why you chose that action, include entity references).

    Constraints:
    - DO NOT ask any follow-up questions.
    - Keep user-facing message <= 25 words.
    - If ticket creation required, include the expected SLA (e.g., "Ticket created — resolution within 48 hours").
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state


    def sim_not_working_node(self, state: CallState) -> CallState:
        """Handles the SIM Not Working scenario."""
        prompt = f"""
    You are a telecom support agent handling a "SIM Not Working" complaint.
    Produce 3 lines (plain text, no questions):
    1) A single, clear user-facing instruction or resolution (one sentence). If a common immediate fix exists, give it (e.g., "Restart phone and reinsert SIM; if still fails, request SIM re-provisioning.").
    2) Internal action label: one of ["remote-provision", "schedule-sim-replacement", "ticket-device-check", "inform-user-no-issue-detected"].
    3) One-line internal diagnostic note referencing extracted entities and confidence.

    Constraints:
    - DO NOT ask follow-up questions.
    - Keep user message short (<=20 words) and deterministic.
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state


    def no_network_coverage_node(self, state: CallState) -> CallState:
        """Handles the No Network Coverage scenario."""
        prompt = f"""
    You are a telecom field-support agent for "No Network Coverage".
    Return exactly 3 lines (plain text):
    1) A single customer-facing message that either explains the cause or gives a decisive next step (e.g., "We will create a ticket for tower inspection; you'll be notified.").
    2) Internal action label: one of ["create-network-ticket","advise-roaming","check-provisioning","no-action"].
    3) One-line internal note with suggested urgency and referenced entities (location, account).

    Constraints:
    - DO NOT ask any follow-up questions.
    - If location is provided in entities, include it in the internal note.
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state


    def internet_speed_slow_node(self, state: CallState) -> CallState:
        """Handles the Internet Speed Slow scenario."""
        prompt = f"""
    You are a telecom troubleshooting agent for "Internet Speed Slow".
    Return exactly 3 lines:
    1) A concise customer-facing resolution or definitive next step (e.g., "We will attempt an automated profile reset; expected improvement within 30 minutes.").
    2) Internal action label: one of ["automated-reset","create-speed-ticket","advise-plan-upgrade","no-action"].
    3) One-line internal diagnostic note (include suggested measurement steps if applicable: speedtest link, time of day, device).

    Constraints:
    - DO NOT ask follow-up questions.
    - Keep customer message <= 25 words and action deterministic.
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state


    def data_not_working_after_recharge_node(self, state: CallState) -> CallState:
        """Handles the Data Not Working After Recharge scenario."""
        prompt = f"""
    You are a support agent for "Data Not Working After Recharge".
    Return exactly 3 lines:
    1) A single, one-sentence user-facing resolution or immediate step (e.g., "We have re-provisioned your data; please restart your device now.").
    2) Internal action label: one of ["reprovision-data","refund-if-failed","open-ticket","no-action"].
    3) One-line internal note referencing recharge_amount/date and whether automatic reprovisioning attempted.

    Constraints:
    - DO NOT ask any follow-up questions.
    - If the entities include a recharge amount/date, reference them in the internal note.
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state


    def call_drops_frequently_node(self, state: CallState) -> CallState:
        """Handles the Call Drops Frequently scenario."""
        prompt = f"""
    You are a network reliability specialist handling "Call Drops Frequently".
    Return exactly 3 lines:
    1) A single customer-facing diagnostic or action (e.g., "We will raise a network investigation ticket; expect update within 48 hours.").
    2) Internal action label: one of ["create-network-investigation","schedule-field-check","check-provisioning","no-action"].
    3) One-line internal note describing probable cause and referencing any location/device entities.

    Constraints:
    - DO NOT ask follow-up questions.
    - Keep user-facing message short and concrete.
    User Input: "{state['clean_text']}"
    Extracted Entities: {state['entities']}
    """
        state['script'] = self.llm.invoke(prompt)
        state['next_action'] = "play_tts"
        return state

    def route_intent_to_node(self, state: CallState) -> str:
        
            
        # Map high confidence intents to the node function names
        if state['intent'] == "Billing Issue":
            return "billing_issue_node"
        elif state['intent'] == "SIM Not Working":
            return "sim_not_working_node"
        elif state['intent'] == "No Network Coverage":
            return "no_network_coverage_node"
        elif state['intent'] == "Internet Speed Slow":
            return "internet_speed_slow_node"
        elif state['intent'] == "Data Not Working After Recharge":
            return "data_not_working_after_recharge_node"
        else: 
            return "call_drops_frequently_node"



