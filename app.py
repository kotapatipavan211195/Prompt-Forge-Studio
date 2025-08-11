"""
app.py — Stable Edit-Then-Run + Code Export
- Session-safe Run button (survives rerun)
- Safe API key loader (.env/env, then st.secrets)
- Logging to outputs/app.log
- Prompt builder → editor → run
- Optional critic/self-review + error-driven fix
- JSON validation
- NEW: Detect code blocks in output, save as scripts, and provide downloads (and zip if multiple)
"""
from __future__ import annotations

import os
import re
import io
import json
import time
import uuid
import zipfile
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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
    st.stop()

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
    st.session_state.chat: List[Dict[str, str]] = []
if "engineered_prompt" not in st.session_state:
    st.session_state.engineered_prompt = ""
if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False  # gate to run on rerun

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
    "python": "py",
    "py": "py",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
    "zsh": "sh",
    "powershell": "ps1",
    "ps1": "ps1",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "json": "json",
    "yaml": "yaml",
    "yml": "yml",
    "toml": "toml",
    "sql": "sql",
    "html": "html",
    "css": "css",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "rust": "rs",
    "go": "go",
    "rb": "rb",
    "ruby": "rb",
    "php": "php",
    "kotlin": "kt",
    "scala": "scala",
    "r": "r",
    "md": "md",
    "markdown": "md",
    "text": "txt",
}

CODE_BLOCK_RE = re.compile(
    r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```",
    re.DOTALL,
)

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Return list of (language, code) for each fenced block found.
    If no language is given, language == ''.
    """
    blocks: List[Tuple[str, str]] = []
    for m in CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        code = m.group(2)
        blocks.append((lang, code))
    return blocks

def save_code_blocks(blocks: List[Tuple[str, str]], base_name: str) -> List[Tuple[str, str, str]]:
    """
    Save code blocks to files under outputs/.
    Returns list of tuples: (language, filename, path)
    """
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

def offer_code_downloads(text: str, base_name: str = "codeblock") -> None:
    """
    Detect code blocks in text, save them, and show download buttons.
    If multiple blocks, also offer a zip of all.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return

    st.markdown("#### Detected code block(s)")
    saved = save_code_blocks(blocks, base_name=base_name)

    # Per-file download buttons
    for lang, filename, path in saved:
        with open(path, "rb") as f:
            st.download_button(
                f"⬇️ Download {filename}",
                data=f.read(),
                file_name=filename,
                key=f"dl-{filename}",
            )

    # Zip if multiple
    if len(saved) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, filename, path in saved:
                zf.write(path, arcname=filename)
        zip_buf.seek(0)
        st.download_button(
            "⬇️ Download all as ZIP",
            data=zip_buf.read(),
            file_name=f"{base_name}.zip",
            key="dl-zip-all",
        )

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
# Input row: instruction → Generate prompt
# ================================================================
user_text = st.chat_input(
    placeholder="Describe what you want (e.g., 'generate test cases', 'draft an email', 'design a schema')."
)

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        st.markdown("**Step 1 — Building a standard prompt...**")
        try:
            engineered = retry(lambda: build_engineered_prompt(user_text), retries=max_retries)
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
        to_save = critic_report or effective  # prefer critic content if present, else the main result
        try:
            out_path = save_text(to_save, prefix="final-output")
            st.download_button("💾 Download result (as text)", data=open(out_path, "rb").read(), file_name=os.path.basename(out_path))
        except Exception as e:
            logger.exception("Failed saving final output: %s", e)
            st.warning("Could not save result to disk.")

        # NEW: detect and export any code blocks
        offer_code_downloads(effective, base_name="script")

        # Record metadata & reset run flag
        st.session_state.last_run_meta = {
            "model": model,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "critic": bool(critic_report),
            "error_fix": bool(fixed_output),
            "conversation_mode": conversation_mode,
        }

    # IMPORTANT: reset flag so we don't re-run forever
    st.session_state.run_requested = False

# ================================================================
# Footer — Diagnostics
# ================================================================
with st.expander("Diagnostics & Last Run", expanded=False):
    st.json(st.session_state.last_run_meta)
    st.caption(f"Logs: `{LOG_FILE}`")

st.caption("Built with Streamlit + OpenAI API. Edit the model in the sidebar.")