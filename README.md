# CriticChain

**AI that audits AI — Multi-Agent Adversarial Review System**

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://github.com/langchain-ai/langgraph)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> A LangGraph-powered quality assurance system where AI agents critique each other to prevent hallucinations and ensure output quality.

---

## ⚡ What CriticChain Is (and Isn't)

**CriticChain is NOT:**
- ❌ A prompt template library
- ❌ A chatbot or assistant
- ❌ A simple code linter
- ❌ A "toy evaluator" for demos

**CriticChain IS:**
- ✅ An **audit system** for AI outputs
- ✅ Built for **professionals already using LLMs in production**
- ✅ Designed to catch what human reviewers miss
- ✅ Your **evidence log** for AI governance compliance

### Features (OSS Edition)

- ✅ **Multi-Agent Workflow**: Router -> Linter -> Analyze -> Critique loop.
- ✅ **Basic Linter**: Catches common syntax errors.
- ✅ **Standard Prompts**: Optimized for general-purpose code/prompt review.
- ✅ **Local Execution**: Runs entirely on your machine via LangGraph.

---

## 🎯 Why CriticChain?

LLMs are powerful, but they lie. They hallucinate. They give you "LGTM" when they should be saying "This is a security risk."

**CriticChain solves this with adversarial review:**

| Problem | CriticChain's Solution |
|---------|----------------------|
| LLM gives "too nice" feedback | **Critic Agent** calls it out: "Your judgment is too lenient" |
| Hallucinations go undetected | **Fact-Check Agent** verifies every claim |
| Syntax errors are missed | **Linter Agent** catches structural issues *before* the LLM sees them |
| No audit trail | Every step is logged with token usage and reasoning |

---

## 🏗️ Architecture

CriticChain orchestrates **11 specialized agents** using LangGraph:

```mermaid
graph TD
    A[Router] --> B[Linter]
    B --> C[Research]
    C --> D{Mode?}
    D -->|Standard| E[Analyze]
    D -->|Hybrid| F[CLI Bypass (Enterprise)]
    D -->|Consensus| E & F
    E --> G[Fact Check]
    G --> H[Draft Review]
    F --> I[Manager (Enterprise)]
    H --> J[Critique]
    I --> J
    J --> K[Refine]
    K --> L[Consistency]
    L --> M[Evaluate]
```

### Agent Roles

| Agent | Role |
|-------|------|
| **Router** | Classifies input (Prompt Engineering vs Programming) |
| **Linter** | Static analysis for structural errors |
| **Research** | Web search for best practices |
| **Analyze** | Deep architectural review |
| **Fact-Check** | Hallucination detection |
| **Draft Review** | Generates initial feedback |
| **Critique** | **The Devil's Advocate** — challenges the draft |
| **Refine** | Incorporates critique feedback |
| **Manager** | Integrates multiple sources (Hybrid/Consensus modes) |
| **Consistency** | Checks for contradictions |
| **Evaluate** | LLM-as-a-Judge scoring |

---

## 🔥 See It In Action

The real proof is in the logs. Here's what happens when **Critique Agent catches Draft Review being too lenient**:

```
[00:39:59] --- Draft Review Node ---
Output: "## PASS ✅ Your prompt demonstrates excellent Chain of Thought design..."

[00:40:46] --- Critique Node ---
Output:
"1. **Your judgment is too lenient.**
   The analysis flagged 'Delimiter Issues: C (Major Improvement Needed)' 
   and 'Potential Bugs', yet you gave a blanket PASS.
   
   Prompt injection risk is a SECURITY VULNERABILITY.
   Revoke the PASS or change to 'Conditional Pass' at minimum.

2. **Your tone is too friendly.**
   Remove phrases like 'Your prompt is at a very high level' and 
   stick to objective facts only.

3. **Lint errors are underemphasized.**
   'Improves readability' is too weak. Say 'Creates bug risk.'"
```

The system then **automatically refined** the review from "PASS" to "Conditional Pass" with specific security warnings.

📁 **Full log file:** [`examples/agent_collaboration_log.txt`](examples/agent_collaboration_log.txt)

---

## 🛡️ What CriticChain Catches (Real Failure Modes)

Without CriticChain, these failures ship to production:

| Failure Mode | What Happens | CriticChain's Catch |
|--------------|--------------|---------------------|
| **Lenient Review** | LLM says "LGTM" to broken code | Critique Agent: "Your judgment is too lenient. This has security risks." |
| **Prompt Injection** | Malicious input bypasses your prompt | Linter Agent flags delimiter issues before LLM processing |
| **Hallucinated Facts** | LLM invents plausible-sounding data | Fact-Check Agent verifies claims against sources |
| **Audit Gap** | No record of AI decisions | Every step logged with tokens, reasoning, and timestamps |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Cloud API key (Gemini)

### Installation

```bash
git clone https://github.com/taka4rest/CriticChain.git
cd CriticChain

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the Web UI

```bash
streamlit run app.py
```

### Run via CLI

```python
from graph import app

result = app.invoke({
    "submission": "Your code or prompt here...",
    "task_type": "Prompt Engineering",  # or "Programming"
    "review_mode": "consensus"  # "standard", "hybrid", or "consensus"
})
```

---

## 📊 Cost Transparency

CriticChain is thorough, which means it uses tokens. Here's what to expect:

| Review Mode | Tokens/Review | Approx. Cost (Gemini 2.5 Pro) |
|-------------|---------------|-------------------------------|
| Standard | ~100,000 | $0.30-0.40 |
| Hybrid | ~80,000 | $0.20-0.30 |
| Consensus | ~130,000 | $0.40-0.50 |

**This is not a "cheap and fast" linter.** It's a deep, multi-perspective analysis system. Use it when quality matters.

---

## 🎓 Origin Story

This system was born in the trenches of **AI education**.

I've spent 28 years as a systems architect, and recently found myself teaching prompt engineering. What I discovered was chaos: students guessing at "magic words", no consistent standards, and AI giving unhelpfully positive feedback.

So I built a review system where:
- The **AI can't be too nice** (Critique Agent won't let it)
- **Syntax errors are caught first** (before expensive LLM calls)
- **Every judgment is auditable** (full logs with reasoning)

While born in education, the architecture proved generic. CriticChain is now a blueprint for Enterprise AI Governance — providing a robust framework designed to detect hallucinations and mitigate data leakage risks before they reach production.

This repository provides the core "Lite Architecture" which is fully functional for individual developers and small teams. It demonstrates the power of the CriticChain workflow.
For Enterprise Users: We also offer an "Enterprise Edition" with advanced auditing logic, multi-agent consensus protocols, and security guardrails. See [Contact](#-contact) for details.

---

## 🔬 How CriticChain Compares

In March 2026, Microsoft launched **Critique** — GPT drafts a response, Claude reviews it. The concept of "AI auditing AI" is now mainstream.

CriticChain takes this further:

| | Microsoft Critique | CriticChain |
|---|---|---|
| **Direction** | One-way (draft → review) | **Iterative dialogue** (Critique ↔ Refine loop) |
| **Agents** | 2 models | **11 specialized agents** |
| **Domain** | Research fact-checking | **Code + Prompt Engineering review** |
| **Deployment** | Cloud-only (Copilot) | **Self-hosted / open source** |
| **Customization** | None | Full control over prompts and workflow |

---

## 🤝 Use Cases

| Domain | Application |
|--------|-------------|
| **Education** | Automated code/prompt review with pedagogical feedback |
| **Enterprise AI** | Quality gate before deploying LLM outputs |
| **Regulated Industries** | Audit trail for AI decisions (EU AI Act compliance) |
| **Development Teams** | PR review augmentation with adversarial checking |

---

## 📬 Contact

Interested in implementing CriticChain for your organization?

Enterprise Support & Consulting: Available for custom implementation and domain-specific tuning.

- GitHub Issues: [taka4rest/CriticChain](https://github.com/taka4rest/CriticChain/issues)
- Enterprise inquiries: Please open an issue with the `enterprise` label or reach out directly.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0-only)**.

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version as a network service, you must make the source code available to users of that service.

**Commercial License:** If your use case is incompatible with AGPL obligations (e.g., embedding CriticChain in a proprietary SaaS product), commercial licenses are available. Contact us for details.

See [LICENSE](LICENSE) for the full license text.

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — The orchestration framework that makes this possible
- [LangChain](https://github.com/langchain-ai/langchain) — Foundation for LLM interactions
- The prompt engineering students who revealed how badly we needed standards
