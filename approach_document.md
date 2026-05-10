# SHL Assessment Recommender — Approach Document

**Dhruv | AI Intern Assignment | SHL Labs | May 2026**

---

## 1. System Design & Architecture

The system is a stateless conversational API built on **FastAPI** with **Groq's hosted llama-3.3-70b-versatile** as the language model. Stateless means the client sends the full conversation history on every `POST /chat` — the server holds no session state. This keeps the backend simple, horizontally scalable, and fully compatible with the evaluation harness.

**Request flow:**

1. Client sends `POST /chat` with the full message history.
2. The server prepends a system prompt (containing all catalog data and behavioral rules) and sends the complete message list to the Groq API in a single call.
3. The model returns a JSON object with `reply`, `recommendations`, and `end_of_conversation`.
4. A verification layer canonicalizes every recommendation against the catalog — fixing minor name drift and dropping any genuine hallucinations.
5. The validated `ChatResponse` is returned to the client.

**Why FastAPI?** Pydantic schema validation at the API boundary catches malformed responses before they reach the client. Built-in OpenAPI docs. Async-ready for future scale.

**Why Groq + llama-3.3-70b?** Groq's inference hardware delivers consistently sub-5-second latency on 70B parameter models — critical for the 30-second per-call timeout. The 128k context window comfortably holds the full 377-item catalog text, the system prompt, conversation history, and generation output with room to spare. Free tier is sufficient for evaluation traffic.

---

## 2. Retrieval Setup — Full Catalog in Context

Rather than a vector database, the entire catalog is injected directly into the system prompt as a compact pipe-delimited table. Each row captures: index, name, test types (keys), duration, remote/adaptive flags, job levels, URL, and a 120-character description excerpt.

```
IDX | NAME | KEYS | DURATION | REMOTE | ADAPTIVE | JOB_LEVELS | URL | DESCRIPTION
0 | OPQ32r | Personality & Behavior | 25 min | R=Y | A=N | Director, Executive… | https://… | Measures 32 workplace behaviour dimensions…
```

**Why no vector search?** The catalog has 377 items. At ~100 tokens per row, the full table is ~40k tokens — well within llama-3.3-70b's 128k context window. Injecting everything means the model can never miss a relevant item due to retrieval error. There is no FAISS warm-up latency, no embedding model to load, and no retrieval-stage recall loss.

**Why include descriptions?** The description is the richest semantic signal for niche queries. Without it, "plant operators at a chemical facility" has no obvious keyword match against "Dependability and Safety Instrument (DSI)". With the description, the model sees "designed to identify employees who will have good dependability and reliability" and correctly selects it.

**When would vector search help?** If the catalog grew beyond ~3,000 items (approaching context limits), a FAISS pre-retrieval step (top-80 candidates → inject into prompt) would reduce prompt size while maintaining recall. For 377 items, it adds latency and recall risk with no benefit.

---

## 3. Prompt Design

The system prompt has four sections:

**3.1 Rules Block** — seven numbered rules in ALL-CAPS headers. Numbered so the model treats them as a checklist: catalog-only constraint, clarify-before-recommend, 1–10 recommendation count, turn cap, off-topic refusal, mid-conversation refinement, and comparison grounding.

**3.2 Output Format Block** — specifies the exact JSON schema. The instruction "ONLY a valid JSON object, no markdown, no code fences" is reinforced in code by `extract_json()`, which strips accidental fences and extracts the first `{...}` blob via regex if the model adds preamble text.

**3.3 Few-Shot Examples** — four examples covering: vague query → clarify, specific query → recommend, off-topic → refuse, mid-conversation edit → update shortlist and close. Each example demonstrates the correct `recommendations: []` vs populated pattern.

**3.4 Catalog Block** — full pipe-delimited table of all 377 assessments.

**Temperature = 0.15** — low enough to make JSON output consistent and reduce hallucination while preserving natural language variation in the `reply` field.

---

## 4. Behavioral Logic

| Behavior | Implementation |
|---|---|
| Clarify vague queries | Rule 2 + few-shot example; model returns `recommendations: []` |
| Turn cap (8 user turns) | Server counts user-role messages; injects SYSTEM NOTE at turn 7; hard-stops at turn 9+ |
| 30-second timeout | Groq's p50 latency is 3–6s on this model; no extra timeout wrapper needed |
| Off-topic refusal | Rule 5 + example; model refuses and returns `[]` |
| Mid-conversation refinement | Full history re-sent each turn; model re-evaluates from scratch naturally |
| Hallucination protection | `verify_recommendations()` — URL set lookup (O(1)) + fuzzy name fallback + hard drop |
| JSON parse failure | Retry once on parse failure before returning a safe fallback response |
| Catalog grounding for comparison | Description column in catalog text gives model factual basis for compare answers |

**Retry logic:** If the model's first response fails JSON parsing (rare at temperature 0.15), the server makes one additional Groq call before returning a graceful fallback. This prevents a single bad generation from failing a turn.

**Fuzzy name matching in `verify_recommendations`:** The model occasionally returns a slightly truncated name (e.g. "OPQ32r" instead of "Occupational Personality Questionnaire OPQ32r"). The verifier tries an exact URL match first, then a fuzzy name match (substring both ways), then drops the item. This recovers valid recommendations that would otherwise be silently discarded.

---

## 5. Evaluation Approach

**Hard evals (must pass):**
- Schema compliance: Pydantic `ChatResponse` enforces field types at the API layer. Malformed LLM output triggers retry then safe fallback — never a 500.
- Catalog-only URLs: `verify_recommendations()` drops anything not in `CATALOG_URLS` (set, O(1) lookup).
- Turn cap: server-side counter, not model-dependent.

**Recall@10:** The full catalog is visible in context on every call, so the model performs semantic matching without a retrieval step. The description field provides the signal needed for non-obvious matches. Temperature 0.15 keeps recommendations stable across turns.

**Behavior probes:**
- *Vague query probe* — "I need an assessment" returns `recommendations: []` (Rule 2 + example).
- *Off-topic probe* — non-assessment queries return refusal (Rule 5 + example).
- *Edit probe* — "Drop X, add Y" is handled naturally because the model re-evaluates the full history, not a cached shortlist.
- *Hallucination probe* — `verify_recommendations()` prevents any invented URL from reaching the response.

**What didn't work:**
- An early version truncated the catalog at 30,000 characters, silently dropping ~100 assessments. Recall@10 on niche queries (healthcare admin, industrial safety) suffered noticeably. Removing the truncation fixed this — full catalog fits comfortably in context.
- A stricter URL-only verifier silently dropped valid recommendations when the model returned a shortened name (no URL match, no fuzzy fallback). Adding the fuzzy name lookup recovered ~15% of valid recs in manual testing against the traces.

---

## 6. Deployment

Deployed on **Render.com** (free tier) as a Python web service. The `Procfile` starts uvicorn bound to Render's dynamic `$PORT`. `GROQ_API_KEY` is stored as a Render environment variable — never in code or committed to Git. `data/catalog.json` is committed to the repository so it is available at runtime without a network fetch on startup.

**Cold-start time:** ~15–20 seconds (Render free tier spin-up). The `/health` endpoint is called first by the evaluator with a 2-minute allowance, after which the service is warm. Subsequent `/chat` calls complete in 3–8 seconds (Groq inference latency), well within the 30-second per-call timeout.

**AI tools used:** GitHub Copilot assisted with boilerplate FastAPI scaffolding. All design decisions, prompt engineering, verification logic, and evaluation methodology were written and understood by hand.
