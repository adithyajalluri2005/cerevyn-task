import os
import json
import uuid
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from groq import Groq

from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.state.state import CallState

load_dotenv()

LOG_DIR = "call_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------- Styles (UI MODIFICATIONS ONLY) -----------------
def _css():
    st.markdown(
        """
        <style>
        /* App-wide */
        :root {
            --muted: #6b7280;
            --accent-1: #06b6d4;
            --accent-2: #fb923c;
            --bg-card: #ffffff;
            --card-border: #e6e6e6;
            --shadow: 0 6px 18px rgba(15,23,42,0.06);
            --rounded: 14px;
        }

        /* Header */
        .app-title { font-size:28px; font-weight:800; margin-bottom:6px; display:flex; align-items:center; gap:10px; }
        .app-sub { color:var(--muted); font-size:13px; margin-top:2px; }

        /* Cards & layout */
        .control-card { padding:14px; border:1px solid var(--card-border); border-radius:var(--rounded); background: var(--bg-card); box-shadow: var(--shadow); }
        .sidebar-card { padding:12px; border-radius:10px; border:1px solid #f1f5f9; background:#fbfbfd; }

        /* Transcript bubbles */
        .transcript-user { background:linear-gradient(90deg,#fff7ed,#fffbf0); padding:12px; border-radius:12px; margin:8px 0; border-left: 6px solid var(--accent-2); }
        .transcript-agent { background:linear-gradient(90deg,#ecfeff,#f4feff); padding:12px; border-radius:12px; margin:8px 0; border-left: 6px solid var(--accent-1); }
        .transcript-system { background:#f3f4f6; padding:10px; border-radius:10px; margin:8px 0; color:#374151; border-left:4px solid #9ca3af; }

        .msg-meta { font-size:12px; color:#475569; display:flex; justify-content:space-between; align-items:center; gap:8px; }
        .msg-text { margin-top:8px; white-space:pre-wrap; font-size:14px; color:#0f172a; }

        /* Snapshot numbers */
        .big-label { font-size:14px; font-weight:700; margin-top:6px; color:white; }
        .big-value { font-size:36px; font-weight:800; line-height:1.05; color:#white; }

        /* Footer */
        .footer-note { font-size:12px; color:#9ca3af; margin-top:8px; }

        /* Small helpers */
        .muted { color:var(--muted); font-size:12px; }
        .pill {
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            font-weight:700;
            font-size:12px;
            border: 1px solid rgba(2,6,23,0.06);
            background: #ffffff;            /* white background for contrast */
            color: #0b1220;                 /* dark text color */
            box-shadow: 0 6px 18px rgba(2,6,23,0.08);
        }

        /* Responsive tweaks */
        @media (max-width: 820px) {
            .big-value { font-size:28px; }
            .app-title { font-size:22px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------- Helpers (unchanged) -----------------
def generate_call_id(prefix: str = "C") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}-{time.strftime('%y%m%d%H%M%S')}"

def speak(text: str) -> None:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 190)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"TTS error: {e}")

def listen() -> str:
    # If no GROQ key, fallback to Google SR
    if not os.getenv("GROQ_API_KEY"):
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as src:
            st.info("🎤 Listening (Google SR)...")
            rec.adjust_for_ambient_noise(src)
            audio = rec.listen(src)
        try:
            return rec.recognize_google(audio)
        except Exception as e:
            return f"(STT error: {e})"

    # Use Groq Whisper
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as src:
            st.info("🎤 Listening (Groq Whisper)...")
            rec.adjust_for_ambient_noise(src)
            audio = rec.listen(src)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            tmp.write(audio.get_wav_data())

        with open(temp_path, "rb") as f:
            transcription_obj = client.audio.transcriptions.create(
                file=f, model="whisper-large-v3", response_format="text", language="en"
            )
        os.unlink(temp_path)
        return str(transcription_obj).strip()
    except Exception as e:
        st.warning(f"Groq STT failed, fallback to Google SR. Error: {e}")
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as src:
            st.info("🎤 Listening (fallback)...")
            rec.adjust_for_ambient_noise(src)
            audio = rec.listen(src)
        try:
            return rec.recognize_google(audio)
        except Exception as e2:
            return f"(STT fallback error: {e2})"

def extract_script_text(script_obj) -> str:
    if script_obj is None:
        return ""
    if isinstance(script_obj, str):
        return script_obj
    if hasattr(script_obj, "content"):
        try:
            return str(script_obj.content)
        except Exception:
            pass
    if isinstance(script_obj, dict):
        for k in ("content", "text", "message"):
            if k in script_obj and isinstance(script_obj[k], str):
                return script_obj[k]
    return str(script_obj)

def json_safe(obj):
    """
    Recursively convert non-JSON-serializable objects into JSON-safe structures.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    name = obj.__class__.__name__
    if hasattr(obj, "content") and name in ("AIMessage", "HumanMessage", "SystemMessage", "ChatMessage"):
        safe = {"type": name, "content": json_safe(getattr(obj, "content", ""))}
        if hasattr(obj, "additional_kwargs"):
            safe["additional_kwargs"] = json_safe(getattr(obj, "additional_kwargs", {}))
        return safe

    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            return json_safe(obj.model_dump())
    except Exception:
        pass

    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return json_safe(dataclasses.asdict(obj))
    except Exception:
        pass

    return str(obj)

def save_call_log(call_id: str, final_state: dict) -> str:
    path = os.path.join(LOG_DIR, f"{call_id}.json")
    safe_state = json_safe(final_state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_state, f, indent=2)
    return path
INTENTS = [
    "Billing Issue",
    "SIM Not Working",
    "No Network Coverage",
    "Internet Speed Slow",
    "Data Not Working After Recharge",
    "Call Drops Frequently",
]
# ----------------- UI (ONLY CHANGES BELOW) -----------------
def load_langgraph_agenticai_app():
    st.set_page_config(page_title="Cerevyn AI — Voice Call Simulator", layout="wide", initial_sidebar_state="expanded")
    _css()

    # Header
    header_col1, header_col2 = st.columns([6,2])
    with header_col1:
        # left: logo + title
        st.markdown(
            '<div style="display:flex; align-items:center; gap:12px;">'
            '<div style="width:48px; height:48px; background:linear-gradient(135deg,#06b6d4,#fb923c); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-weight:800;">C</div>'
            '<div>'
            '<div class="app-title">🎧 Cerevyn AI — Voice Call</div>'
            '<div class="app-sub">Voice-only web call with STT, intent detection, scripted response, and TTS.</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with header_col2:
        st.markdown('<div style="text-align:right;"><span class="muted">Call ID</span><div class="pill" style="margin-top:6px;">{}</div></div>'.format(st.session_state.get('call_id','-')), unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar (minimal settings but reorganized)
    with st.sidebar:
        st.markdown("### ⚙️ Settings & Tools")
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY not set — STT will fallback to Google.")
            pasted = st.text_input("GROQ_API_KEY", type="password")
            if pasted:
                os.environ["GROQ_API_KEY"] = pasted
                st.experimental_rerun()

        # Model selection grouped visually
        st.markdown("**Model**")
        model_name = st.selectbox("LLM Model", ["openai/gpt-oss-20b"], index=0)
        enable_tts = st.checkbox("Enable TTS (pyttsx3)", value=True)
        st.markdown("---")

        st.markdown("**Intents**")
        # Render intents as pill badges using the CSS 'pill' class already present
        intents_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;'>"
        for it in INTENTS:
            intents_html += f"<div class='pill' title='{it}' style='font-size:12px; padding:6px 10px;'>{it}</div>"
        intents_html += "</div>"
        st.markdown(intents_html, unsafe_allow_html=True)

        st.markdown("**Call Tools**")
        if st.button("🔄 New Call ID (sidebar)"):
            st.session_state['call_id'] = generate_call_id()
            st.session_state['transcript'] = []
            st.session_state['last_state'] = None
            st.experimental_rerun()

        # Quick housekeeping
        if st.button("🧹 Clear Transcript"):
            st.session_state['transcript'] = []
            st.success("Transcript cleared.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="muted">Tip: Press <strong>Start Call</strong>, speak, and wait for the agent to respond. Use downloads to export logs.</div>', unsafe_allow_html=True)

    # Init session state
    if 'call_id' not in st.session_state:
        st.session_state['call_id'] = generate_call_id()
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = []
    if 'last_state' not in st.session_state:
        st.session_state['last_state'] = None

    # Main layout: left conversation, right dashboard
    left, right = st.columns([3,1])

    with left:
        # Controls row (Start/End + quick export)
        control_row = st.container()
        c1, c2, c3 = control_row.columns([1,1,1])
        start_call = c1.button("▶️ Start Call", use_container_width=True, type="primary")
        end_call = c2.button("⏹ End Call", use_container_width=True)
        if c3.button("⬇️ Export Transcript", use_container_width=True):
            transcript_text = "\n\n".join([f"{'Caller' if m['speaker']=='user' else ('Agent' if m['speaker']=='agent' else 'System')} [{time.strftime('%H:%M:%S', time.localtime(m['ts']))}]\n{m['text']}" for m in st.session_state['transcript']])
            st.download_button("Download .txt", data=transcript_text, file_name=f"{st.session_state['call_id']}_transcript.txt", mime="text/plain")

        st.markdown("---")
        st.markdown("### Conversation")
        st.info("Press Start Call and speak when prompted. The agent will respond with TTS.")

        # Start call flow (unchanged functional logic)
        if start_call:
            with st.spinner('Listening...'):
                user_text = listen()

            if not user_text or user_text.startswith('('):
                st.error(f"Voice capture failed: {user_text or 'empty input'}")
            else:
                st.session_state['transcript'].append(
                    {"speaker":"user","text":user_text,"ts":time.time()}
                )

                # Build model + graph
                try:
                    model = GroqLLM(model=model_name).get_llm_model()
                    gb = GraphBuilder(model)
                    gb.call_center_build_graph()
                    app = gb.setup_graph()
                except Exception as e:
                    st.error(f"Graph init failed: {e}")
                    app = None

                init_state: CallState = {
                    "call_id": st.session_state['call_id'],
                    "transcript": st.session_state['transcript'],
                    "clean_text": "",
                    "intent": "",
                    "confidence": 0.0,
                    "entities": {},
                    "script": "",
                    "next_action": "end_call",
                    "test_input": None,
                }

                if app is not None:
                    with st.spinner('Processing...'):
                        final_state = app.invoke(init_state)
                else:
                    final_state = {
                        **init_state,
                        "script": "(System error) Unable to process request.",
                        "intent": "error",
                        "confidence": 0.0,
                    }

                # Extract + speak; also normalize script for logging
                script_text = extract_script_text(final_state.get('script',''))
                final_state['script'] = script_text
                if script_text:
                    st.session_state['transcript'].append(
                        {"speaker":"agent","text":script_text,"ts":time.time()}
                    )
                    if enable_tts:
                        speak(script_text)

                # Save full final state and keep for download/view
                saved_path = save_call_log(st.session_state['call_id'], final_state)
                st.session_state['last_state'] = {"path": saved_path, "state": json_safe(final_state)}

        # End call
        if end_call:
            st.session_state['transcript'].append({"speaker":"system","text":"Call ended.","ts":time.time()})
            st.success("Call ended.")

        # Transcript viewer (with updated styles and small controls)
        st.markdown("---")
        st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'><h4>Transcript</h4><div class='muted'>Latest messages shown first</div></div>", unsafe_allow_html=True)

        if st.session_state['transcript']:
            # show latest first but keep ordering in session_state unchanged
            for msg in reversed(st.session_state['transcript'][-200:]):  # show up to 200 recent messages
                ts = time.strftime('%H:%M:%S', time.localtime(msg['ts']))
                meta_html = f"<div class='msg-meta'><div><strong>{'Caller' if msg['speaker']=='user' else ('Agent' if msg['speaker']=='agent' else 'System')}</strong> • {ts}</div><div><small class='muted'>{msg.get('meta','')}</small></div></div>"
                if msg['speaker'] == 'user':
                    st.markdown(f"<div class='transcript-user'>{meta_html}<div class='msg-text'>{msg['text']}</div></div>", unsafe_allow_html=True)
                elif msg['speaker'] == 'agent':
                    st.markdown(f"<div class='transcript-agent'>{meta_html}<div class='msg-text'>{msg['text']}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='transcript-system'>{meta_html}<div class='msg-text'>{msg['text']}</div></div>", unsafe_allow_html=True)
        else:
            st.info('No messages yet — start a call to populate the transcript.')

    with right:
        st.markdown("### Snapshot")
        last = st.session_state.get('last_state')
        if last:
            state = last['state']  # already JSON-safe
            intent = state.get('intent','-')
            conf = float(state.get('confidence',0.0) or 0.0)
            st.markdown("<div class='big-label'>Intent</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-value'>{intent}</div>", unsafe_allow_html=True)
            st.markdown("<div class='big-label'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-value'>{conf:.2f}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.download_button(
                '⬇️ Download Final State',
                data=json.dumps(state, indent=2),
                file_name=os.path.basename(last['path']),
                mime='application/json',
                use_container_width=True
            )
        else:
            st.info('No calls yet — start a call to see results.')

        st.markdown('---')
        st.markdown('### Call Logs')

        # Load recent logs and list them as markdown (no boxes)
        logs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
        if logs:
            bulleted = "\n".join([f"- {name}" for name in logs[:50]])
            st.markdown(bulleted or "- (no logs)")
            selected_log = st.selectbox("Select a log to view (JSON)", logs, index=0 if logs else None)
            if selected_log:
                fpath = os.path.join(LOG_DIR, selected_log)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Show JSON of selected log
                    st.json(data)
                    st.download_button(
                        f"⬇️ Download {selected_log}",
                        data=json.dumps(data, indent=2),
                        file_name=selected_log,
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to load log: {e}")
        else:
            st.write('No saved logs yet.')

        st.markdown('---')
        st.markdown('<div class="footer-note">Run: <code>streamlit run app.py</code></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    load_langgraph_agenticai_app()
