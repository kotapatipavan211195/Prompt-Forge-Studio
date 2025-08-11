# Prompt Forge Studio (Streamlit)

**Build → Edit → Run** LLM prompts with best-practice prompt engineering, optional self-review, error-driven fixes, and automatic **code export**.

---

## Table of Contents
- [What this app does](#what-this-app-does)
- [Requirements](#requirements)
- [1. Create an OpenAI account & API key](#1-create-an-openai-account--api-key)
- [2. Choose how you want to provide your API key](#2-choose-how-you-want-to-provide-your-api-key)
  - [Option A: Environment variable (recommended for local dev)](#option-a-environment-variable-recommended-for-local-dev)
  - [Option B: `.env` file (easy, pairs with `python-dotenv`)](#option-b-env-file-easy-pairs-with-python-dotenv)
  - [Option C: Streamlit `secrets.toml` (great for Streamlit Cloud)](#option-c-streamlit-secretstoml-great-for-streamlit-cloud)
- [3. Verify your API key works](#3-verify-your-api-key-works)
- [4. Install and run the app](#4-install-and-run-the-app)
- [5. How to use the app](#5-how-to-use-the-app)
- [Troubleshooting](#troubleshooting)
- [Security best practices](#security-best-practices)
- [Common questions (FAQ)](#common-questions-faq)
- [Repo layout](#repo-layout)
- [License](#license)

---

## What this app does
- **Two-step flow:** Type a short instruction → app generates a **standardized engineered prompt** → you **edit** it → click **Run** to execute.
- **Prompt engineering standards:** Role, Task, Context, Inputs, Constraints, Steps, Examples, Output, Tone, Acceptance Criteria.
- **Conversation-aware** execution (optional).
- **Critic/self-review** pass and **error-driven fix** mode (paste tracebacks/failing tests).
- **Code export:** auto-detects fenced code blocks in results and offers per-file downloads and a ZIP.
- **Logging & artifacts:** saves prompts/results/logs in `outputs/`.

---

## Requirements
- **Python 3.9+** (3.10–3.12 recommended)
- **OpenAI API key** with active billing
- Internet access (the app calls OpenAI’s API)
- OS: macOS, Linux, or Windows

---

## 1. Create an OpenAI account & API key
1. Sign in to the **OpenAI Platform**.
2. Go to **API Keys** (Profile → View API keys → Create new secret key).
3. Click **Create new secret key** and copy it. Keys start with `sk-...`.
4. Ensure **billing** is set up; some accounts require adding a payment method before requests succeed.

> Keep this key private. Anyone with your key can use your quota.

---

## 2. Choose how you want to provide your API key

You can provide the key in any of these ways. The app loads keys in this order:
1) Environment variable → 2) `.env` → 3) Streamlit `secrets.toml`.

### Option A: Environment variable (recommended for local dev)
**macOS/Linux (bash/zsh):**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Windows PowerShell:**
```powershell
setx OPENAI_API_KEY "sk-your-key-here"
# Open a NEW terminal after this so the variable is available
```
**You can temporarily set it just for one session:**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

---

### Option B: .env file (easy, pairs with python-dotenv)
1.	Copy the example:

```bash
cp .env.example .env
```
2.	Edit .env:
```env
OPENAI_API_KEY=sk-your-key-here
```
3.  The app automatically loads .env.

---

### Option C: Streamlit secrets.toml (great for Streamlit Cloud)
1.  Create:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
2.  Edit .streamlit/secrets.toml:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
**Avoid mistakes:**
1.  Use straight quotes ", not smart quotes.
2.  Don’t leave it blank like OPENAI_API_KEY =.
3.  No trailing characters.

On Streamlit Cloud, set this via Secrets in the app settings.

---

## 3. Verify your API key works
```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(resp.choices[0].message.content)
```
If this prints a reply, your key is valid.

---

## 4. Install and run the app
```bash
git clone https://github.com/yourname/prompt-forge-studio.git
cd prompt-forge-studio

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 5. How to use the app
1.	Enter a short instruction in the input box.
2.	App generates a standardized prompt.
3.	Edit the prompt.
4.	Click Run with this prompt.
5.	If output contains fenced code blocks:
    - You’ll see Download buttons for each file.
    - if multiple blocks: ZIP download available.
6.	Optional: paste an error/traceback in the error box to trigger fixes.

---

## Troubleshooting

**Missing OPENAI_API_KEY**
  - Key not set or file not read.

**StreamlitSecretNotFoundError parsing secrets file**
  - Malformed TOML. Fix syntax.

**Run button does nothing**
  - Make sure engineered prompt exists before running.
  - Check outputs/app.log.

**Model errors**
  - Try another model in the sidebar.

**Invalid JSON output**
  - Adjust prompt or change output format.

**Proxy issues**
  - Configure network/proxy properly.

---

## Security best practices
1. 	Never commit API keys to Git.
2. 	Rotate keys if exposed.
3. 	Use env vars for production.
4. 	Avoid plaintext storage on shared machines.

---

## Common questions (FAQ)
**Where does the app save files?**
-  outputs/ folder.

**Can I change the default model?**
-  Yes, via sidebar.

**How does code export work?**
-  Detects fenced code blocks, infers extension, saves, and provides download links.

**Does the app stream responses?**
-  Yes, toggle in sidebar.

**Can I deploy it?**
-  Yes, Streamlit Cloud or any Python server with Streamlit.

---

## Repo layout
```
Prompt-Forge-Studio/
├── app.py                  # Streamlit app (main file)
├── .env
├── .streamlit/
│   └── secrets.toml.example
├── logs/
│   └── llm_app.log         # Application log file
├── output/
│   └── .gitkeep            # Output Python script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE
```

## License

MIT License
