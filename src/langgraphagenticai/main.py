import os
import json
import uuid
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
import pyttsx3
from groq import Groq
from streamlit_mic_recorder import mic_recorder

from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.state.state import CallState

# Optional Supabase (disabled if no key)
try:
    from supabase import create_client
except Exception:
    create_client = None

load_dotenv()

LOG_DIR = "call_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _css():
    st.markdown(
        """
        <style>
        .app-title { font-size:28px; font-weight:700; margin-bottom:6px; }
        .muted { color:#6b7280; font-size:12px }
        /* Updated backgrounds and text colors */
        .transcript-user { background:#fff7ed; padding:10px; border-radius:10px; margin:6px 0; border-left: 4px solid #fb923c; color:#7c2d12; }
        .transcript-user strong { color:#9a3412; }
        .transcript-agent { background:#ecfeff; padding:10px; border-radius:10px; margin:6px 0; border-left: 4px solid #06b6d4; color:#0e7490; }
        .transcript-agent strong { color:#155e75; }
        .footer-note { font-size:12px; color:#9ca3af; }
        .big-label { font-size:18px; font-weight:700; margin-top:6px; }
        .big-value { font-size:40px; font-weight:800; line-height:1.1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def generate_call_id(prefix: str = "C") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}-{time.strftime('%y%m%d%H%M%S')}"

def init_tts():
    TTS_ENGINE = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 190)
        engine.setProperty("volume", 1.0)
        TTS_ENGINE = engine
    except Exception as e:
        if not st.session_state.get("_tts_unavailable_warned", False):
            st.warning(f"TTS unavailable on this host ({e}). Falling back to text-only.")
            st.session_state["_tts_unavailable_warned"] = True
    return TTS_ENGINE


def speak(text: str) -> None:
    engine = init_tts()
    if not engine:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        if not st.session_state.get("_tts_failed_warned", False):
            st.warning(f"TTS failed: {e}. Falling back to text-only.")
            st.session_state["_tts_failed_warned"] = True

def transcribe_bytes_wav(wav_bytes: bytes) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "(STT error: GROQ_API_KEY not set)"
    try:
        client = Groq(api_key=api_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            temp_path = tmp.name
        with open(temp_path, "rb") as f:
            transcription_obj = client.audio.transcriptions.create(
                file=f, model="whisper-large-v3", response_format="text", language="en"
            )
        os.unlink(temp_path)
        return str(transcription_obj).strip()
    except Exception as e:
        return f"(STT error: {e})"

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
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_state, f, indent=2)
    except Exception as e:
        try:
            st.warning(f"Failed to write local log file: {e}")
        except Exception:
            pass
    # Optional Supabase insert
    sb = get_supabase_client()
    if sb:
        try:
            table = os.getenv("SUPABASE_TABLE") or "call_states"
            row = {
                "call_id": call_id,
                "final_state": safe_state,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            try:
                sb.table(table).upsert(row, on_conflict="call_id").execute()
            except Exception:
                sb.table(table).insert(row).execute()
        except Exception as e:
            try:
                st.warning(f"Supabase insert failed: {e}")
            except Exception:
                pass
    return path

def load_langgraph_agenticai_app():
    st.set_page_config(page_title="Cerevyn AI — Voice Call", layout="wide")
    _css()

    # Header
    header_col1, header_col2 = st.columns([6,2])
    with header_col1:
        st.markdown('<div class="app-title">🎧 Cerevyn AI — Voice Call</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Voice-only web call with in-browser recording, STT, intent, script, and TTS.</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown(f"**Call ID**\n\n`{st.session_state.get('call_id','-')}`")

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set. STT requires Groq Whisper.")
            pasted = st.text_input("GROQ_API_KEY", type="password")
            if pasted:
                os.environ["GROQ_API_KEY"] = pasted
                st.rerun()
        model_name = st.selectbox("LLM Model", ["openai/gpt-oss-20b"], index=0)
        enable_tts = st.checkbox("Enable TTS (server-side)", value=True)
        st.markdown("---")
        if st.button("🔄 New Call ID (sidebar)"):
            st.session_state['call_id'] = generate_call_id()
            st.session_state['transcript'] = []
            st.session_state['last_state'] = None
            st.rerun()

    # Session init
    if 'call_id' not in st.session_state:
        st.session_state['call_id'] = generate_call_id()
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = []
    if 'last_state' not in st.session_state:
        st.session_state['last_state'] = None

    left, right = st.columns([3,1])

    with left:
        st.markdown("### Conversation")
        st.info("Click Start Call to begin recording. Click End Call to stop. Audio is processed on submit.")

        # In-browser mic recording (no PyAudio)
        audio = mic_recorder(
            start_prompt="▶️ Start Call",
            stop_prompt="⏹ End Call",
            just_once=False,
            key="mic",
            format="wav",
        )

        # When recording finished, audio is a dict like {"bytes": b"...", "sample_rate": 16000}
        if audio and isinstance(audio, dict) and audio.get("bytes"):
            wav_bytes = audio["bytes"]
            with st.spinner("Transcribing..."):
                user_text = transcribe_bytes_wav(wav_bytes)

            if not user_text or user_text.startswith("("):
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

                # Normalize script and speak
                script_text = extract_script_text(final_state.get('script',''))
                final_state['script'] = script_text
                if script_text:
                    st.session_state['transcript'].append(
                        {"speaker":"agent","text":script_text,"ts":time.time()}
                    )
                    if enable_tts:
                        speak(script_text)

                # Save full final state
                saved_path = save_call_log(st.session_state['call_id'], final_state)
                st.session_state['last_state'] = {"path": saved_path, "state": json_safe(final_state)}

        # Transcript
        st.markdown("---")
        if st.session_state['transcript']:
            for msg in st.session_state['transcript']:
                ts = time.strftime('%H:%M:%S', time.localtime(msg['ts']))
                if msg['speaker'] == 'user':
                    st.markdown(f"<div class='transcript-user'><strong>Caller • {ts}</strong><div style='margin-top:6px'>{msg['text']}</div></div>", unsafe_allow_html=True)
                elif msg['speaker'] == 'agent':
                    st.markdown(f"<div class='transcript-agent'><strong>Agent • {ts}</strong><div style='margin-top:6px'>{msg['text']}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='transcript-user'><strong>System • {ts}</strong><div style='margin-top:6px'>{msg['text']}</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Snapshot")
        last = st.session_state.get('last_state')
        if last:
            state = last['state']
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
        logs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
        if logs:
            selected_log = st.selectbox("Select a log to view (JSON)", logs, index=0)
            if selected_log:
                fpath = os.path.join(LOG_DIR, selected_log)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
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