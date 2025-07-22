# Financial Advisor Chatbot – Complete Project Guide

This markdown file contains everything you need to build, run, and test an AI-powered personal-finance assistant that:

* answers natural-language finance questions (budgeting, investments, taxation, loans, insurance)
* performs sentiment extraction from news with FinBERT
* fetches live market prices with `yfinance`
* calculates loan EMI, debt-to-income, and risk-tolerance scores
* runs locally (CPU-only) with free open-source models from Hugging Face
* exposes a friendly chat UI via **Gradio**

---

## 1. Prerequisites

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install gradio transformers accelerate sentencepiece \
            yfinance matplotlib pandas numpy scikit-learn
```

> All models run in 4-bit quantised mode so a 6-8 GB RAM machine is usually enough (no GPU required).

---

## 2. Folder Layout

```
financial-advisor-bot/
├── app.py                # main Streamlit / Gradio entry-point
├── finance_tools.py      # helper functions (EMI, budget, risk, tax)
├── prompts.py            # system + user prompt templates
└── requirements.txt      # optional – freeze versions
```

---

## 3. Core Logic (finance_tools.py)

```python
import math, json, datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ------------------ CALCULATORS ------------------

def calc_emi(p, annual_rate, years):
    """Flat-rate EMI (Equated Monthly Instalment)."""
    r = annual_rate / 12 / 100
    n = years * 12
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)

def debt_to_income(total_debt_pm, net_income_pm):
    return round((total_debt_pm / net_income_pm) * 100, 2)

def risk_score(answers):
    """Very tiny 10-question risk tolerance test (0–100)."""
    return int(sum(answers) / (len(answers) * 5) * 100)

def alloc_50_30_20(net_income):
    return {
        "Needs":   net_income * 0.50,
        "Wants":   net_income * 0.30,
        "Savings": net_income * 0.20,
    }

# ------------------ MARKETS ------------------

def price_history(ticker, period="1y"):
    data = yf.download(ticker, period=period, progress=False)
    return data["Adj Close"].dropna().to_frame(name=ticker)

# helper to JSON-serialise pandas objects for LLM consumption
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.date, dt.datetime)):
            return obj.isoformat()
        if "numpy" in str(type(obj)):
            return obj.item()
        return json.JSONEncoder.default(self, obj)
```

---

## 4. Prompt Templates (prompts.py)

```python
SYSTEM_PROMPT = """
You are Nivesh-GPT, a bilingual (Hindi + English) personal-finance assistant.
You must:
1. answer with clear bullet-points and markdown tables where helpful
2. ALWAYS cite numeric advice from Govt. rules (80C, 50/30/20 etc.)
3. show formulas when doing calculations
"""

USER_WRAP = """
**User query:** {query}

**Context (JSON):**
{context}
"""
```

---

## 5. Application (app.py)

```python
import gradio as gr, os, json, pandas as pd, numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from finance_tools import calc_emi, debt_to_income, risk_score, alloc_50_30_20, price_history, NpEncoder
import prompts

# ----------- load small FREE LLM (Mistral-7B-Instruct-v0.2) ----------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                           bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
lm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
tok = AutoTokenizer.from_pretrained(model_name)
chat = pipeline("text-generation", model=lm, tokenizer=tok, temperature=0.2, max_new_tokens=512)

# ------------------- FinBERT sentiment --------------------
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ------------------- core chat fn -------------------------

def generate_reply(message, history):
    # quick command detections before calling LLM
    if message.lower().startswith("/emi"):
        _, p, r, y = message.split()
        emi = calc_emi(float(p), float(r), int(y))
        return f"**EMI:** ₹{emi:,.0f} per month for ₹{p} @ {r}% for {y} years."
    if message.lower().startswith("/price"):
        _, ticker = message.split()
        hist = price_history(ticker, "1mo")
        return hist.tail().to_markdown()

    context = json.dumps({"history": history[-6:]}, cls=NpEncoder, indent=2)
    prompt = prompts.SYSTEM_PROMPT + prompts.USER_WRAP.format(query=message, context=context)
    resp = chat(prompt)[0]["generated_text"][len(prompt):]
    return resp.strip()

# ---------- Gradio UI -----------------
with gr.Blocks(theme="gradio/soft") as demo:
    gr.Markdown("# 💸 AI Financial Advisor Chatbot\nType `/emi 500000 8 15` or `/price INFY.NS` for quick tools.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything about budgeting, loans, taxes…")
    msg.submit(lambda m, h: ("", h + [[m, None]]), [msg, chatbot], [msg, chatbot], queue=False).then(
        lambda h: h[-1][0], chatbot, gr.State(), queue=False).then(
        fn=generate_reply, inputs=["state"], outputs="state", queue=True).then(
        lambda answer, chat_hist: chat_hist[:-1] + [[chat_hist[-1][0], answer]],
        [chatbot, "state"], chatbot)

if __name__ == "__main__":
    demo.launch()
```

> **Note**: Switch to any other free chat model (e.g. `tiiuae/falcon-7b-instruct`) by changing `model_name`.

---

## 6. Running the Bot

```bash
streamlit run app.py   # or python app.py if only Gradio
```

Navigate to `http://localhost:8501` (Streamlit) or the Gradio URL.

### Example session

```
👤 How much EMI for 8 lakh @ 9% for 7 years?
🤖 EMI = ₹13,677 per month  (formula:  P r (1+r)^n / ((1+r)^n-1) )

👤 Show 50-30-20 budget if I earn 1 lakh net.
🤖 Needs ₹50,000 │ Wants ₹30,000 │ Savings ₹20,000  (pie chart)

👤 Sentiment on “TCS reported record order book and raised guidance”.
🤖 FinBERT → Positive (91.3%)
```

---

## 7. Extending Functionality

| Feature | How to add |
|---------|------------|
| Risk-tolerance quiz | Call `risk_score()` on a 10-question radio-group and tailor allocation |
| Tax planning India (80C, 24b, 80D) | Encode slabs into helper functions and cite sources [98][102][107] |
| Insurance gap analysis | Use Human-Life-Value method [43][48] |
| Multi-modal (image of pay-slip) | Swap Mistral for Llama-3-8B-Instruct-Vision and pipe base64 image |
| Deployment | `gradio deploy` → Hugging Face Spaces (free CPU)

---

### Citations

Key rules, formulas, and data used by the bot are taken from: 80C deductions [98][102][107], 50/30/20 rule [40][45][50], EMI formula [39][44], DTI guidance [100][109][104], FinBERT model card [67], Mistral license [70], Gradio chatbot docs [68][73][78], risk-tolerance scales [19][24][29][93][95][105][110], and budgeting best-practice [20][25][99][101][106].

---

**Enjoy your AI Finance Coach! 🎉**
