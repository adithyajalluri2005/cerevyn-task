# Cerevyn AI — Voice Call Center Agent

A Streamlit app that turns your browser into a voice call center agent:
- In-browser recording (no PyAudio)
- Speech-to-Text with Groq Whisper
- Deterministic intent detection via LangGraph (6 fixed intents)
- Browser Text-to-Speech (Web Speech API)
- Per-call IDs, JSON logs, and call history viewer

## Features

- Voice capture in browser using streamlit-mic-recorder
- STT: Groq Whisper (model: whisper-large-v3)
- Graph orchestration with LangGraph:
  - preprocess → nlu → domain node → end
  - Strict NLU: always one of 6 intents
- Confidence shown as 1–100 (internally 0.0–1.0)
- Browser TTS (no server audio deps)
- Start/End Call buttons; a new Call ID is generated on Start
- De-duplication: each recording processed once (prevents repeated agent replies)
- JSON logging to call_logs/ with in-app viewer and download
- Optional Supabase persistence (auto-disabled without URL+Key)

## Architecture

- UI entry: app.py (runs the Streamlit app)
- App logic/UI: src/langgraphagenticai/main.py
- Graph builder: src/langgraphagenticai/graph/graph_builder.py
- Nodes (business logic): src/langgraphagenticai/nodes/nodes.py
- State models: src/langgraphagenticai/state/state.py
- LLM adapter: src/langgraphagenticai/LLMS/groqllm.py

Graph flow:
1) preprocess_node — sanitize and extract last user utterance
2) nlu_node — structured NLU (AllowedIntent Literal) + keyword fallback
3) domain node — one of the 6 intent handlers generates concise script
4) end

Allowed intents:
- Billing Issue
- SIM Not Working
- No Network Coverage
- Internet Speed Slow
- Data Not Working After Recharge
- Call Drops Frequently

## Requirements

- Python 3.10–3.13
- No system audio libraries needed (PyAudio not used)
- Browser with mic permissions

Install Python deps:
```
pip install -r requirements.txt
```

Minimal requirements (already in requirements.txt):
```
streamlit==1.51.0
python-dotenv==1.2.1
groq==0.33.0
langchain-core==1.0.5
langgraph==1.0.3
streamlit-mic-recorder==0.0.8
```

Optional for alternative TTS (server-generated mp3):
- Add gtts==2.5.4 and use the gTTS path in code.

## Configuration

Set your Groq API key (required for STT):
- Local .env
  - Create .env (do NOT commit it) with:
    ```
    GROQ_API_KEY=your_groq_key
    ```
- Or Streamlit Secrets (Cloud)
  - In app settings → Secrets:
    ```
    GROQ_API_KEY="your_groq_key"
    ```

Note: .gitignore already ignores .env.

## Run

Local:
```
streamlit run app.py
```
If app.py imports the app from src, ensure paths are correct.

Cloud (Streamlit Community Cloud):
- Push repo (ensure no secrets committed)
- Set GROQ_API_KEY in Secrets
- Deploy; app command: streamlit run app.py

## Usage

1) Open the app.
2) In sidebar, set GROQ_API_KEY if not detected.
3) Click “Start Call” — a new Call ID is created.
4) Press and hold “Hold to Record” (from streamlit-mic-recorder), release to send.
5) The app:
   - Transcribes with Groq Whisper
   - Classifies intent into one of the six intents
   - Generates a concise agent script
   - Speaks via browser TTS (if enabled)
6) Repeat to add more exchanges. Click “End Call” to finish.
7) Download the final call state (JSON) or browse history.

## Confidence

- Internally stored as 0.0–1.0
- Displayed as 1–100 for readability
- Progress bar uses the 0.0–1.0 value

## Extending

Add a new intent:
1) Update AllowedIntent in state.py and ALLOWED_INTENTS in nodes.py
2) Add keywords to INTENT_KEYWORDS in nodes.py for fallback
3) Implement a new domain node in nodes.py (e.g., def roaming_issue_node)
4) Register the node and routing in graph_builder.py
5) Re-deploy

## Troubleshooting

- Push blocked by “secrets found”:
  - Remove .env from git history and rotate keys
  - Keep .env in .gitignore, commit a .env.example instead
- Microphone not recording:
  - Allow mic permissions in the browser
  - Some mobile browsers restrict inline audio
- TTS doesn’t play:
  - Browser TTS may be blocked by autoplay; ensure the tab has user interaction
- Progress bar error:
  - It expects 0..1; the app converts the 1–100 display back to 0..1 for the bar
- Repeated answers:
  - The app tracks last_audio_id; ensure you’re using the latest code

## Security

- Do not commit secrets (.env is ignored)
- Rotate exposed keys immediately
- Logs in call_logs/ may contain PII; handle according to your policy

## Project Scripts

- Start dev server:
  ```
  streamlit run app.py
  ```
- Freeze deps (pip-tools users):
  ```
  pip freeze > requirements.txt
  ```

## License

Add your license here (e.g., MIT).

## Acknowledgements

- Streamlit
- LangGraph
- LangChain
- Groq Whisper
- streamlit-mic-recorder