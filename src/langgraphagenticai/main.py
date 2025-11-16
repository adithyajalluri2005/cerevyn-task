import os
import json
import uuid
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import streamlit.components.v1 as components

from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.state.state import CallState

# Optional Supabase (disabled if not installed)
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
        .app-title { 
            font-size: 32px; 
            font-weight: 800; 
            margin-bottom: 8px;
            background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .muted { color: #6b7280; font-size: 14px; line-height: 1.6; }
        .call-id-box {
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            font-weight: 700;
            color: #1e293b;
        }
        .transcript-user { 
            background: #fff7ed; 
            padding: 14px; 
            border-radius: 12px; 
            margin: 8px 0; 
            border-left: 5px solid #fb923c; 
            color: #7c2d12;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .transcript-user strong { color: #9a3412; font-size: 13px; }
        .transcript-agent { 
            background: #ecfeff; 
            padding: 14px; 
            border-radius: 12px; 
            margin: 8px 0; 
            border-left: 5px solid #06b6d4; 
            color: #0e7490;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .transcript-agent strong { color: #155e75; font-size: 13px; }
        .footer-note { font-size: 11px; color: #9ca3af; margin-top: 16px; }
        .big-label { 
            font-size: 14px; 
            font-weight: 600; 
            margin-top: 12px; 
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .big-value { 
            font-size: 48px; 
            font-weight: 900; 
            line-height: 1.1;
            color: #0f172a;
        }
        .section-header {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #1e293b;
        }
        .info-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 12px;
            border-radius: 8px;
            color: #1e40af;
            font-size: 14px;
            margin-bottom: 16px;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def generate_call_id(prefix: str = "C") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}-{time.strftime('%y%m%d%H%M%S')}"

def speak(text: str) -> None:
    if not text:
        return
    escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    html_code = f"""
    <script>
        const utterance = new SpeechSynthesisUtterance('{escaped_text}');
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(html_code, height=0)

def transcribe_bytes_wav(wav_bytes: bytes) -> str:
    api_key =st.secrets["GROQ_API_KEY"]
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
    return path

def load_langgraph_agenticai_app():
    st.set_page_config(page_title="Cerevyn AI — Voice Call Center", layout="wide", initial_sidebar_state="expanded")
    _css()

    # Header
    st.markdown('<div class="app-title">🎧 Cerevyn AI — Voice Call Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Intelligent voice call system with real-time STT, intent detection, scripted responses, and browser-based TTS.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # API Key input
        if not st.secrets["GROQ_API_KEY"]:
            st.error("⚠️ GROQ_API_KEY not set")
            pasted = st.text_input("Enter GROQ_API_KEY", type="password", key="api_key_input")
            if pasted:
                os.environ["GROQ_API_KEY"] = pasted
                st.success("✅ API Key set!")
                st.rerun()
        else:
            st.success("✅ GROQ API Key configured")
        
        st.markdown("---")
        
        # Model selection
        model_name = st.selectbox("🤖 LLM Model", ["openai/gpt-oss-20b"], index=0)
        enable_tts = st.checkbox("🔊 Enable TTS (Browser)", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Session Info")
        st.info(f"**Active Call ID:**\n`{st.session_state.get('call_id', 'None')}`")
        st.info(f"**Total Exchanges:**\n{len(st.session_state.get('transcript', [])) // 2}")

    # Session state initialization
    if 'call_id' not in st.session_state:
        st.session_state['call_id'] = None
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = []
    if 'last_state' not in st.session_state:
        st.session_state['last_state'] = None
    if 'call_active' not in st.session_state:
        st.session_state['call_active'] = False
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False

    # Main layout
    left, right = st.columns([2.5, 1.5])

    with left:
        st.markdown('<div class="section-header">📞 Call Interface</div>', unsafe_allow_html=True)
        
        # Call controls row
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            start_call_btn = st.button("▶️ Start Call", use_container_width=True, type="primary", disabled=st.session_state.get('call_active', False))
        
        with col2:
            end_call_btn = st.button("⏹ End Call", use_container_width=True, disabled=not st.session_state.get('call_active', False))
        
        with col3:
            if st.session_state.get('call_id'):
                st.markdown(f'<div class="call-id-box">{st.session_state["call_id"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="call-id-box">No Active Call</div>', unsafe_allow_html=True)

        # Start call logic
        if start_call_btn:
            # Generate new call ID
            st.session_state['call_id'] = generate_call_id()
            st.session_state['transcript'] = []
            st.session_state['last_state'] = None
            st.session_state['call_active'] = True
            st.success(f"✅ Call started: {st.session_state['call_id']}")
            st.rerun()

        # End call logic
        if end_call_btn:
            st.session_state['call_active'] = False
            st.session_state['transcript'].append({"speaker": "system", "text": "Call ended by user.", "ts": time.time()})
            st.warning(f"⏹ Call ended: {st.session_state['call_id']}")
            st.rerun()

        st.markdown("---")

        # Recording interface (only if call is active)
        if st.session_state.get('call_active', False):
            st.markdown('<div class="info-box">🎙️ Press the button below to record your message. Release to process.</div>', unsafe_allow_html=True)
            
            audio = mic_recorder(
                start_prompt="🎤 Hold to Record",
                stop_prompt="⏸ Release to Send",
                just_once=False,
                key="mic",
                format="wav",
            )

            if audio and isinstance(audio, dict) and audio.get("bytes"):
                wav_bytes = audio["bytes"]
                
                with st.spinner("🔄 Transcribing..."):
                    user_text = transcribe_bytes_wav(wav_bytes)

                if not user_text or user_text.startswith("("):
                    st.error(f"❌ Voice capture failed: {user_text or 'empty input'}")
                else:
                    st.session_state['transcript'].append(
                        {"speaker": "user", "text": user_text, "ts": time.time()}
                    )

                    # Build model + graph
                    try:
                        model = GroqLLM(model=model_name).get_llm_model()
                        gb = GraphBuilder(model)
                        gb.call_center_build_graph()
                        app = gb.setup_graph()
                    except Exception as e:
                        st.error(f"❌ Graph init failed: {e}")
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
                        with st.spinner('🤖 Processing intent...'):
                            final_state = app.invoke(init_state)
                    else:
                        final_state = {
                            **init_state,
                            "script": "(System error) Unable to process request.",
                            "intent": "error",
                            "confidence": 0.0,
                        }

                    # Extract response and speak
                    script_text = extract_script_text(final_state.get('script', ''))
                    final_state['script'] = script_text
                    
                    if script_text:
                        st.session_state['transcript'].append(
                            {"speaker": "agent", "text": script_text, "ts": time.time()}
                        )
                        if enable_tts:
                            speak(script_text)

                    # Save call log
                    saved_path = save_call_log(st.session_state['call_id'], final_state)
                    st.session_state['last_state'] = {"path": saved_path, "state": json_safe(final_state)}
                    
                    st.rerun()
        else:
            st.info("👆 Click **Start Call** to begin a new conversation")

        # Transcript display
        st.markdown("---")
        st.markdown('<div class="section-header">💬 Conversation Transcript</div>', unsafe_allow_html=True)
        
        if st.session_state['transcript']:
            for msg in st.session_state['transcript']:
                ts = time.strftime('%H:%M:%S', time.localtime(msg['ts']))
                if msg['speaker'] == 'user':
                    st.markdown(
                        f"<div class='transcript-user'><strong>👤 CALLER • {ts}</strong>"
                        f"<div style='margin-top:8px; font-size:15px;'>{msg['text']}</div></div>",
                        unsafe_allow_html=True
                    )
                elif msg['speaker'] == 'agent':
                    st.markdown(
                        f"<div class='transcript-agent'><strong>🤖 AGENT • {ts}</strong>"
                        f"<div style='margin-top:8px; font-size:15px;'>{msg['text']}</div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='transcript-user'><strong>⚙️ SYSTEM • {ts}</strong>"
                        f"<div style='margin-top:8px; font-size:14px;'>{msg['text']}</div></div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No messages yet. Start a call to begin.")

    with right:
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        last = st.session_state.get('last_state')
        if last:
            state = last['state']
            intent = state.get('intent', 'Unknown')
            conf = float(state.get('confidence', 0.0) or 0.0)
            
            st.markdown("<div class='big-label'>Detected Intent</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-value'>{intent}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='big-label'>Confidence Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-value'>{conf:.2f}</div>", unsafe_allow_html=True)
            
            st.progress(conf)

            st.markdown("---")
            
            safe_filename = os.path.basename(last.get('path') or f"{st.session_state.get('call_id', 'call')}.json")
            st.download_button(
                '⬇️ Download Call State',
                data=json.dumps(state, indent=2),
                file_name=safe_filename,
                mime='application/json',
                use_container_width=True
            )
        else:
            st.info('📭 No call data yet.\n\nStart a call to see analytics.')

        st.markdown('---')
        st.markdown('<div class="section-header">📂 Call History</div>', unsafe_allow_html=True)
        
        logs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json')], reverse=True)
        if logs:
            selected_log = st.selectbox("Select a call log", logs, index=0, key="log_selector")
            
            if selected_log:
                fpath = os.path.join(LOG_DIR, selected_log)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    with st.expander("📄 View JSON", expanded=False):
                        st.json(data)
                    
                    st.download_button(
                        f"⬇️ Download {selected_log}",
                        data=json.dumps(data, indent=2),
                        file_name=selected_log,
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl-{selected_log}"
                    )
                except Exception as e:
                    st.error(f"Failed to load log: {e}")
        else:
            st.write('📭 No saved logs yet.')

        st.markdown('<div class="footer-note">Powered by Cerevyn AI • Built with Streamlit</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    load_langgraph_agenticai_app()