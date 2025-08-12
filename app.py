from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable, Tuple
from dotenv import load_dotenv
from openai import OpenAI
# Removed eager audio imports to prevent errors when packages absent
# from audio_recorder_streamlit import audio_recorder  # NEW 🎤
# from streamlit_mic_recorder import mic_recorder
from base64 import b64decode
import hashlib  # ADDED: for deduping audio saves
import os
import re
import io
import json
import time
import uuid
import zipfile
import logging
import streamlit as st  # ADDED
import streamlit.components.v1 as components  # ADDED

# ================================================================
# Page / Logging / Outputs
# ================================================================
st.set_page_config(page_title="LLM Prompt Builder → Executor", page_icon="🤖", layout="wide")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("prompt_app")

# ================================================================
# API Key & Client (safe loader)
# ================================================================
load_dotenv()  # loads .env if present
api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("Missing OPENAI_API_KEY. Set it in your environment, .env, or .streamlit/secrets.toml")
    st.stop()  # FIX: keep stop inside conditional

client = OpenAI(api_key=api_key)

# ================================================================
# Prompt Engineering Standards
# ================================================================
PROMPT_ENGINEERING_GUIDELINES = (
    """
Follow these standards when generating an engineered prompt:
1) ROLE — Expert persona & tooling context (if any).
2) TASK — Precise, unambiguous restatement of the user goal.
3) CONTEXT — Helpful background, glossary, assumptions.
4) INPUTS — Enumerate provided inputs; list missing inputs + assumptions.
5) CONSTRAINTS — Scope, style, time/tokens, do/don't rules, guardrails.
6) STEPS — Ordered, reproducible plan to complete the task.
7) EXAMPLES — 1–2 minimal examples (if enabled) of desired behavior.
8) OUTPUT — Exact expected structure/format; schema if JSON.
9) TONE/STYLE — Desired tone, reading level, formatting.
10) QUALITY BAR — Acceptance criteria or self-checks.
"""
).strip()

# ================================================================
# Session State
# ================================================================
if "chat" not in st.session_state:
    st.session_state.chat = []  # removed inline type comment to satisfy linter
if "engineered_prompt" not in st.session_state:
    st.session_state.engineered_prompt = ""
if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False
# NEW: voice → text pipeline
if "pending_instruction" not in st.session_state:
    st.session_state.pending_instruction = ""   # set from mic transcript when user clicks “Use...”
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""      # last transcribed text
# NEW: audio dedupe caches
if "saved_audio_hashes" not in st.session_state:
    st.session_state.saved_audio_hashes = {}  # sha256 -> path
if "last_composer_audio_hash" not in st.session_state:
    st.session_state.last_composer_audio_hash = None

# ================================================================
# Sidebar Controls
# ================================================================
st.sidebar.header("⚙️ Settings")
model = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini", help="e.g., gpt-4o-mini, gpt-4.1-mini, o3-mini")
stream_tokens = st.sidebar.toggle("Stream responses", value=True)
max_retries = st.sidebar.slider("Max retries (API)", 0, 5, 2)

st.sidebar.subheader("Prompt Builder Options")
tone = st.sidebar.selectbox("Tone", ["neutral", "friendly", "professional", "concise", "enthusiastic"], index=0)
output_format = st.sidebar.selectbox("Output format", ["plain text", "markdown", "bullet list", "JSON"], index=1)
include_examples = st.sidebar.toggle("Include small example(s)", value=True)
include_criteria = st.sidebar.toggle("Include acceptance criteria", value=True)
force_json_validation = st.sidebar.toggle("Validate JSON output (if JSON)", value=True)

st.sidebar.subheader("Conversation & Code Quality")
conversation_mode = st.sidebar.toggle("Keep conversation context (multi-turn)", value=True)
code_self_review = st.sidebar.toggle("Self-review code answers (critic + fix)", value=True)
error_driven_fix = st.sidebar.toggle("Enable error-driven fix mode", value=True)

with st.sidebar.expander("Prompt Engineering Standards", expanded=False):
    st.markdown(PROMPT_ENGINEERING_GUIDELINES)

# ================================================================
# System Prompts
# ================================================================
PROMPT_BUILDER_SYSTEM = f"""
You are an excellent prompt engineer. Given a user's short instruction, create an execution-ready prompt for a capable model.
Apply these standards:\n{PROMPT_ENGINEERING_GUIDELINES}\n
Return only the engineered prompt text. Use these sections in this order:
- **Role**
- **Task**
- **Context**
- **Inputs**
- **Constraints**
- **Steps**
- **Examples** (include only if helpful and permitted)
- **Output** (format: {output_format})
- **Tone** ({tone})
- **Acceptance Criteria** (include if requested)
""".strip()

EXECUTOR_SYSTEM = """
You are a capable assistant. Read the engineered prompt and produce the final result exactly as requested under **Output** and **Tone**.
If the engineered prompt is ambiguous, make sensible assumptions and proceed.
For coding tasks:
- Prefer correct, runnable code; include minimal examples/tests.
- Avoid placeholders unless necessary, and clearly mark them.
- Cite language/tool versions if relevant.
""".strip()

CODE_CRITIC_SYSTEM = """
You are a senior code reviewer. Given the original instruction and the assistant's output, identify issues (correctness, runtime risks, security, deps, edge cases).
Return: (1) concise findings; (2) corrected version; (3) brief rationale. If output isn't code, critique clarity/structure.
""".strip()

ERROR_FIXER_SYSTEM = """
You are a debugging assistant. Given the original instruction, the previous answer, and an error/traceback or failing test, provide a diagnosis and a corrected solution. Prefer minimal changes unless a rewrite is safer.
""".strip()

# ================================================================
# Helpers (saving, retries, json)
# ================================================================
def save_text(text: str, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"{prefix}-{ts}-{uuid.uuid4().hex[:8]}.txt"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved %s (%d bytes)", path, len(text))
    return path

def save_bytes(b: bytes, prefix: str, ext: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"{prefix}-{ts}-{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "wb") as f:
        f.write(b)
    logger.info("Saved %s (%d bytes)", path, len(b))
    return path

# ADDED: helper to avoid saving same audio multiple times per Streamlit reruns
def save_audio_once(audio_bytes: bytes, *, prefix: str = "voice", ext: str = "wav") -> str:
    h = hashlib.sha256(audio_bytes).hexdigest()[:32]
    existing = st.session_state.saved_audio_hashes.get(h)
    if existing:
        logger.debug("Audio already saved (hash=%s) -> %s", h, existing)
        return existing
    path = save_bytes(audio_bytes, prefix=prefix, ext=ext)
    st.session_state.saved_audio_hashes[h] = path
    logger.info("Audio saved once (hash=%s) -> %s", h, path)
    return path

def retry(fn: Callable[[], str], *, retries: int = 2, base_delay: float = 0.8) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait = base_delay * (2 ** attempt)
            logger.warning("Attempt %d failed: %s (waiting %.2fs)", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"All retries failed: {last_err}")

def validate_json_output(text: str) -> Optional[str]:
    looks_json = output_format.upper() == "JSON" or text.strip().startswith(("{", "["))
    if not looks_json:
        return None
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("JSON validation failed: %s", e)
        return None

# ================================================================
# LLM compatibility layer: Responses API → fallback Chat Completions
# ================================================================
def _call_via_responses(sys: str, user: str, stream: bool) -> str:
    if stream:
        acc: List[str] = []
        placeholder = st.empty()
        with client.responses.stream(
            model=model,
            input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        ) as stream_resp:
            for event in stream_resp:
                if event.type == "response.output_text.delta":
                    acc.append(event.delta)
                    placeholder.markdown("".join(acc))
            _ = stream_resp.get_final_response()
        out = "".join(acc).strip()
        placeholder.markdown(out)
        return out

    resp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
    )
    out = getattr(resp, "output_text", None)
    if out is not None:
        return out.strip()
    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    chunks.append(getattr(c, "text", ""))
    return "".join(chunks).strip()

def _call_via_chat(sys: str, user: str, stream: bool) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        stream=False,
    )
    return resp.choices[0].message.content.strip()

def llm_once(sys: str, user: str, *, stream: bool) -> str:
    text = user.strip()
    if not text:
        raise ValueError("Empty input to model.")
    logger.info("LLM call (stream=%s) len=%d", stream, len(text))
    try:
        return _call_via_responses(sys, user, stream)
    except Exception as e:
        logger.warning("Responses API failed: %s; falling back to Chat Completions", e)
        return _call_via_chat(sys, user, False)

# ================================================================
# Builder / Executor / Critic / Fix
# ================================================================
def build_engineered_prompt(user_instruction: str) -> str:
    bullets = [
        f"Tone: {tone}",
        f"Desired output format: {output_format}",
        "Include 1–2 short examples" if include_examples else "No examples",
        "Include acceptance criteria" if include_criteria else "No acceptance criteria",
    ]
    builder_user = (
        "Create an execution-ready prompt from the instruction below, using the standards and sections specified.\n\n"
        + "\n".join(f"- {b}" for b in bullets)
        + "\n\n[Instruction]\n"
        + user_instruction
    )
    return llm_once(PROMPT_BUILDER_SYSTEM, builder_user, stream=False)

def execute_engineered_prompt(engineered_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    if conversation_mode and history:
        recent = history[-6:]
        transcript = "\n".join(f"[{m.get('role', 'user')}] {m.get('content','')}" for m in recent)
        engineered_with_context = (
            "Context from recent conversation (for continuity):\n"
            + transcript
            + "\n\n---\n\n"
            + engineered_prompt
        )
        return llm_once(EXECUTOR_SYSTEM, engineered_with_context, stream=stream_tokens)
    return llm_once(EXECUTOR_SYSTEM, engineered_prompt, stream=stream_tokens)

def run_code_self_review(original_instruction: str, assistant_output: str) -> str:
    critic_input = (
        "[Original Instruction]\n"
        + original_instruction.strip()
        + "\n\n[Assistant Output to Review]\n"
        + assistant_output.strip()
    )
    return llm_once(CODE_CRITIC_SYSTEM, critic_input, stream=False)

def run_error_driven_fix(original_instruction: str, previous_answer: str, error_text: str) -> str:
    fix_input = (
        "[Original Instruction]\n"
        + original_instruction.strip()
        + "\n\n[Previous Answer]\n"
        + previous_answer.strip()
        + "\n\n[Error / Traceback / Test Failure]\n"
        + error_text.strip()
    )
    return llm_once(ERROR_FIXER_SYSTEM, fix_input, stream=stream_tokens)

# ================================================================
# NEW — Code block extraction + downloads
# ================================================================
LANG_EXT = {
    "python": "py","py": "py","bash": "sh","sh": "sh","shell": "sh","zsh": "sh",
    "powershell": "ps1","ps1": "ps1","javascript": "js","js": "js",
    "typescript": "ts","ts": "ts","json": "json","yaml": "yaml","yml": "yml",
    "toml": "toml","sql": "sql","html": "html","css": "css","java": "java",
    "c": "c","cpp": "cpp","c++": "cpp","rust": "rs","go": "go","rb": "rb",
    "ruby": "rb","php": "php","kotlin": "kt","scala": "scala","r": "r",
    "md": "md","markdown": "md","text": "txt",
}
CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for m in CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        code = m.group(2)
        blocks.append((lang, code))
    return blocks

def save_code_blocks(blocks: List[Tuple[str, str]], base_name: str) -> List[Tuple[str, str, str]]:
    saved: List[Tuple[str, str, str]] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for idx, (lang, code) in enumerate(blocks, start=1):
        ext = LANG_EXT.get(lang, "txt")
        filename = f"{base_name}-{ts}-{idx}.{ext}"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        saved.append((lang or "plain", filename, path))
        logger.info("Saved code block: %s (%s)", path, lang or "plain")
    return saved

def offer_code_downloads(text: str, base_name: str = "script") -> None:
    """
    Detect code blocks and show download buttons without writing to disk.
    Only when the user clicks a button do we (optionally) persist to outputs/.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return

    st.markdown("#### Detected code block(s)")

    # Build suggested filenames (no disk writes yet)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suggested = []
    for idx, (lang, code) in enumerate(blocks, start=1):
        ext = LANG_EXT.get((lang or "").lower(), "txt")
        fname = f"{base_name}-{ts}-{idx}.{ext}"
        suggested.append((lang or "plain", code, fname))

    # Per-file download buttons (in-memory)
    for idx, (lang, code, fname) in enumerate(suggested, start=1):
        clicked = st.download_button(
            label=f"⬇️ Download {fname}",
            data=code.encode("utf-8"),
            file_name=fname,
            key=f"dl-{fname}",
        )
        if clicked:
            # Persist because the user explicitly downloaded
            try:
                path = os.path.join(OUTPUT_DIR, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                logger.info("Saved code file after download click: %s", path)
            except Exception as e:
                logger.exception("Failed to persist %s after download: %s", fname, e)

    # ZIP (in-memory)
    if len(suggested) > 1:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, code, fname in suggested:
                zf.writestr(fname, code)
        buf.seek(0)
        clicked_zip = st.download_button(
            label="⬇️ Download all as ZIP",
            data=buf.getvalue(),
            file_name=f"{base_name}.zip",
            key="dl-zip-all",
        )
        if clicked_zip:
            # Optional: persist the zip after click
            try:
                zip_path = os.path.join(OUTPUT_DIR, f"{base_name}-{ts}.zip")
                with open(zip_path, "wb") as f:
                    f.write(buf.getvalue())
                logger.info("Saved ZIP after download click: %s", zip_path)
            except Exception as e:
                logger.exception("Failed to persist ZIP after download: %s", e)
# ================================================================
# NEW — Microphone → Transcription → Instruction
# ================================================================
def transcribe_wav_file(path: str) -> str:
    """
    Send a WAV file to OpenAI transcription and return text.
    Uses whisper-1 for broad compatibility.
    """
    try:
        with open(path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = getattr(tr, "text", "").strip()
        return text
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        raise

# Try both mic components; we'll use whichever works in the user’s browser
# Try both recorders; we’ll use whichever works
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_ARS = True
except Exception:
    HAS_ARS = False

try:
    from streamlit_mic_recorder import mic_recorder
    HAS_SMR = True
except Exception:
    HAS_SMR = False

def b64_wav_to_bytes(b64_or_dict):
    if isinstance(b64_or_dict, dict):
        b64_data = b64_or_dict.get("bytes") or b64_or_dict.get("audio")
    else:
        b64_data = b64_or_dict
    if not b64_data:
        return None
    if isinstance(b64_data, str) and "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    return b64decode(b64_data)

with st.expander("🎤 Voice to instruction (optional)", expanded=False):
    st.caption("Record your voice or upload an audio file; we’ll transcribe it and you can use it as your instruction.")

    # --- Mini diagnostics (runs in the browser) ---
    with st.popover("Mic diagnostics"):
        st.write("Checks your browser permission state for microphone.")
        components.html(
            """
            <script>
            async function run() {
              const out = { https: location.protocol === 'https:' || location.hostname === 'localhost' };
              try {
                const perm = await navigator.permissions.query({name:'microphone'});
                out.permission = perm.state;
              } catch(e) { out.permission = 'unsupported (' + e + ')'; }
              try {
                out.mediaDevices = !!navigator.mediaDevices;
              } catch(e) { out.mediaDevices = false; }
              const pre = document.createElement('pre');
              pre.textContent = JSON.stringify(out, null, 2);
              document.body.appendChild(pre);
            }
            run();
            </script>
            """,
            height=120,
        )

    audio_bytes = None
    source_used = None

    # --- Primary recorder
    if HAS_ARS:
        try:
            audio_bytes = audio_recorder(
                text="Click to record / stop",
                recording_color="#e8f0fe",
                neutral_color="#f0f0f0",
                icon_size="2x",
                pause_threshold=1.2,
                sample_rate=16000,
            )
            if audio_bytes:
                source_used = "audio-recorder-streamlit"
        except Exception as e:
            logger.warning("audio-recorder-streamlit render error: %s", e)

    # --- Fallback recorder
    if not audio_bytes and HAS_SMR:
        try:
            rec = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop",
                just_once=False,
                use_container_width=True,
                format="wav",
                key="mic_fallback",
            )
            if rec:
                audio_bytes = b64_wav_to_bytes(rec)
                if audio_bytes:
                    source_used = "streamlit-mic-recorder"
        except Exception as e:
            logger.warning("streamlit-mic-recorder render error: %s", e)

    st.caption(f"Recorder status: {'OK ('+source_used+')' if audio_bytes else 'no audio captured yet'}")

    # --- NEW: Upload fallback (works everywhere)
    up = st.file_uploader("Or upload audio (wav, m4a, mp3)", type=["wav", "m4a", "mp3"])
    if up is not None and not audio_bytes:
        audio_bytes = up.read()
        source_used = f"upload:{up.type or up.name}"

    if audio_bytes:
        ext = "wav"
        if source_used and "mp3" in source_used: ext = "mp3"
        if source_used and "m4a" in source_used: ext = "m4a"
        # CHANGED: use save_audio_once instead of save_bytes (dedupe)
        wav_path = save_audio_once(audio_bytes, prefix="voice", ext=ext)
        st.audio(audio_bytes, format=f"audio/{ext}")
        if st.button("📝 Transcribe", key="btn_transcribe"):
            with st.spinner("Transcribing…"):
                try:
                    t = transcribe_wav_file(wav_path)  # your existing function
                    st.session_state.voice_transcript = t
                    st.success("Transcription complete.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

    if st.session_state.voice_transcript:
        st.text_area("Transcript", value=st.session_state.voice_transcript, height=120, key="voice_transcript_area")
        if st.button("➡️ Use this as my instruction", key="btn_use_transcript"):
            st.session_state.pending_instruction = st.session_state.voice_transcript
            st.success("Transcript queued as your next instruction.")

# ================================================================
# UI — Header & History
# ================================================================
left, right = st.columns([0.72, 0.28])
with left:
    st.title("🤖 Prompt Builder → Executor")
    st.caption("Give a short instruction → get an engineered prompt → edit → click Run to execute.")
with right:
    st.info("Model, tone, and output format are adjustable in the sidebar.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================================================================
# Input row: chat input OR pending instruction from voice
# ================================================================
# --- Single-mic ChatGPT-style composer (one mic on the left) ---

# Ensure these helpers/imports exist near top of file:
# from audio_recorder_streamlit import audio_recorder
# def save_bytes(...):  # already in your file
# def transcribe_wav_file(path: str) -> str:  # already in your file

if "compose_text" not in st.session_state:
    st.session_state.compose_text = ""

def _send_compose():
    txt = st.session_state.get("compose_text", "").strip()
    if txt:
        st.session_state["__pending_user_text"] = txt
        st.session_state.compose_text = ""  # clear after send

composer = st.container()
with composer:
    # One clean row: [ mic ] [ text input ] [ send ]
    col_mic, col_text, col_send = st.columns([0.08, 0.74, 0.18], vertical_alignment="center")

    # ---- Single mic button (audio-recorder-streamlit only)
    with col_mic:
        try:
            ar_bytes = audio_recorder(
                text="",                    # no caption next to icon
                recording_color="#334155",  # darker while recording
                neutral_color="#1f2937",    # dark idle
                icon_size="1x",
                pause_threshold=1.1,
                sample_rate=16000,
                key="composer_ars_single",
            )
            if ar_bytes:
                h = hashlib.sha256(ar_bytes).hexdigest()[:32]
                if h != st.session_state.last_composer_audio_hash:
                    wav_path = save_audio_once(ar_bytes, prefix="voice", ext="wav")
                    with st.spinner("Transcribing…"):
                        try:
                            t = transcribe_wav_file(wav_path)
                            # append transcript to input text
                            st.session_state.compose_text = (st.session_state.compose_text + " " + t).strip()
                            st.session_state.last_composer_audio_hash = h
                            st.toast("Transcript added to the input.", icon="🎙️")
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                else:
                    logger.debug("Duplicate composer audio hash detected; skipping re-save & re-transcribe")
        except Exception as e:
            logger.warning("Mic failed: %s", e)
            st.caption("🎙️ unavailable")

    # ---- Text input (no value= to avoid Streamlit warning)
    with col_text:
        st.text_input(
            "Instruction",
            key="compose_text",
            label_visibility="collapsed",
            placeholder="Describe what you want me to do…",
        )

    # ---- Send button
    with col_send:
        st.button("➤ Send", type="primary", use_container_width=True, on_click=_send_compose)

# Consume any pending text (like st.chat_input would return)
user_text = st.session_state.pop("__pending_user_text", "")

# If the user clicked “Use this as my instruction”, feed it into the flow once
if not user_text and st.session_state.pending_instruction:
    user_text = st.session_state.pending_instruction
    st.session_state.pending_instruction = ""  # consume it once

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        st.markdown("**Step 1 — Building a standard prompt...**")
        try:
            def _call(): return build_engineered_prompt(user_text)
            engineered = retry(_call, retries=max_retries)
        except Exception as e:
            st.error(f"Prompt builder failed: {e}")
            logger.exception("Builder failed")
            engineered = "[ERROR] Prompt builder failed. Please revise the instruction and try again."
    st.session_state.engineered_prompt = engineered
    st.session_state.run_requested = False  # reset any stale run flag

# ================================================================
# Editor: show whenever we have an engineered prompt in session
# ================================================================
if st.session_state.engineered_prompt:
    with st.chat_message("assistant"):
        with st.expander("🔧 Engineered prompt (you can edit)", expanded=True):
            st.session_state.engineered_prompt = st.text_area(
                "Engineered prompt",
                value=st.session_state.engineered_prompt,
                height=320,
                label_visibility="collapsed",
                key="engineered_prompt_area",
            )
            c1, c2 = st.columns([0.55, 0.45])
            with c1:
                def _set_run():
                    st.session_state.run_requested = True
                st.button("▶️ Run with this prompt", type="primary", use_container_width=True, on_click=_set_run)
            with c2:
                try:
                    prompt_path = save_text(st.session_state.engineered_prompt, prefix="engineered-prompt")
                    st.download_button(
                        "💾 Download prompt",
                        data=open(prompt_path, "rb").read(),
                        file_name=os.path.basename(prompt_path),
                        use_container_width=True,
                    )
                except Exception as e:
                    logger.exception("Failed saving engineered prompt: %s", e)
                    st.warning("Could not save engineered prompt.")

    # Optional error box for error-driven fixes
    error_text = st.text_area(
        "Optional: Paste error/traceback or failing test output for debugging",
        height=120,
        key="error_box",
    )
else:
    error_text = ""

# ================================================================
# Step 2 — Execute if run_requested is set (survives rerun)
# ================================================================
if st.session_state.run_requested:
    with st.chat_message("assistant"):
        st.markdown("**Step 2 — Executing the prompt...**")
        try:
            final_output = retry(
                lambda: execute_engineered_prompt(st.session_state.engineered_prompt, history=st.session_state.chat),
                retries=max_retries,
            )
        except Exception as e:
            st.error(f"Execution failed: {e}")
            logger.exception("Execution failed")
            final_output = "[ERROR] Execution failed. Please adjust the prompt and retry."

        # Optional critic pass
        critic_report: Optional[str] = None
        if code_self_review:
            with st.spinner("Running self-review (critic pass)..."):
                try:
                    critic_report = retry(
                        lambda: run_code_self_review(st.session_state.chat[-1]["content"], final_output),
                        retries=max_retries,
                    )
                except Exception as e:
                    logger.exception("Critic pass failed: %s", e)
                    critic_report = None

        # Error-driven fix if error text provided
        fixed_output: Optional[str] = None
        if error_driven_fix and error_text and error_text.strip():
            with st.spinner("Applying error-driven fix..."):
                try:
                    fixed_output = retry(
                        lambda: run_error_driven_fix(st.session_state.chat[-1]["content"], final_output, error_text),
                        retries=max_retries,
                    )
                except Exception as e:
                    logger.exception("Error-driven fix failed: %s", e)
                    fixed_output = None

        # Optional JSON validation
        pretty_json: Optional[str] = None
        if force_json_validation:
            pretty_json = validate_json_output(fixed_output or final_output)
            if output_format.upper() == "JSON" and pretty_json is None:
                st.warning("Expected JSON but could not parse. Refine the prompt or review the response.")

        # Present results
        st.markdown("**Result:**")
        effective = fixed_output or final_output
        if pretty_json is not None:
            st.code(pretty_json, language="json")
        else:
            st.markdown(effective)

        if critic_report:
            st.markdown("---")
            st.markdown("**Self-Review (critic report & proposed fix):**")
            st.markdown(critic_report)
        if fixed_output:
            st.markdown("---")
            st.markdown("**Error-Driven Corrected Result:**")
            st.markdown(fixed_output)

        # Save artifact and offer downloads
        to_save = critic_report or effective
        try:
            out_path = save_text(to_save, prefix="final-output")
            st.download_button("💾 Download result (as text)", data=open(out_path, "rb").read(), file_name=os.path.basename(out_path))
        except Exception as e:
            logger.exception("Failed saving final output: %s", e)
            st.warning("Could not save result to disk.")

        # Detect & export code blocks
        offer_code_downloads(effective, base_name="script")

        # Record metadata & reset run flag
        st.session_state.last_run_meta = {
            "model": model,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "critic": bool(critic_report),
            "error_fix": bool(fixed_output),
            "conversation_mode": conversation_mode,
        }

    st.session_state.run_requested = False

# ================================================================
# Footer — Diagnostics
# ================================================================
with st.expander("Diagnostics & Last Run", expanded=False):
    st.json(st.session_state.last_run_meta)
    st.caption(f"Logs: `{LOG_FILE}`")

st.caption("Built with Streamlit + OpenAI API. Voice input powered by audio-recorder-streamlit + whisper-1.")
# ================================================================
# End of app.py