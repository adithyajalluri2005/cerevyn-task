import time
from src.langgraphagenticai.state.state import NLUOutput, CallState, TranscriptEntry
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
import pyttsx3
import speech_recognition as sr
from groq import Groq
import tempfile
import os

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
        intents = [
                "Billing Issue", "SIM Not Working", "No Network Coverage",
                "Internet Speed Slow", "Data Not Working After Recharge",
                "Call Drops Frequently"
            ]
        # Strong system-style instructions + few-shot examples + strict JSON-only requirement.
        prompt = f"""
    You are an NLU module for a telecom call center. Your job:
    1) Classify the user's intent into exactly one of these canonical intents: {intents}.
    2) Extract relevant entities as canonical key:value pairs (use keys like account_number, subscriber_id, date, time, recharge_amount, plan_name, location, device_model, error_code).
    3) Return a single JSON object ONLY that matches this schema:
    {{
        "intent": "<one of the canonical intents>",
        "confidence": <float between 0.0 and 1.0>,
        "entities": {{ <entity_key>: "<string or standardized format>" }},
        "notes": "<short internal note, optional>"
    }}

    Do NOT return any explanatory text or markdown—only valid JSON.and "confidence" to a low value (e.g., 0.05). Normalize phone/account numbers by removing non-digit characters. For dates use ISO format YYYY-MM-DD when possible.
    Dont respond with Unknowmn intent. Always pick the best matching intent from the list.

    Examples (input -> output):
    1) "My bill is 3500 and I think they overcharged my account 9876543210" ->
    {{ "intent":"Billing Issue", "confidence":0.95,
        "entities":{{"account_number":"9876543210","amount":"3500"}},
        "notes":"possible overcharge complaint" }}
    2) "After recharging ₹199 my data still doesn't work since 01-06-2025" ->
    {{ "intent":"Data Not Working After Recharge","confidence":0.98,
        "entities":{{"recharge_amount":"199","date":"2025-06-01"}},
        "notes":"fresh recharge reported" }}
    3) "I keep getting dropped calls in my area" ->
    {{ "intent":"Call Drops Frequently","confidence":0.90,
        "entities":{{"location":"user_reported_area"}},
        "notes":"user-reported area instability" }}

    User Input: "{state['clean_text']}"
    """

        structured_llm = self.llm.with_structured_output(NLUOutput)

        try:
            nlu_result: NLUOutput = structured_llm.invoke(prompt)
            state['intent'] = nlu_result.intent
            state['confidence'] = nlu_result.confidence
            state['entities'] = nlu_result.entities
        except Exception as e:
            print(f"NLU Structured output failed: {e}")
            state['intent'] = "Unknown"
            state['confidence'] = 0.0
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



