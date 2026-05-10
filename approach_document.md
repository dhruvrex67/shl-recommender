# SHL Assessment Recommender — Approach Document
**Dhruv | AI Intern Assignment | SHL Labs | May 2026**

---

## 1. System Design & Architecture

The system is a stateless conversational API built on **FastAPI** with **Gemini 1.5 Flash** as the language model backbone. Stateless means the client sends the full conversation history on every request — the server holds no session memory. This keeps the backend simple, horizontally scalable, and fully compatible with the evaluation harness.

**Request flow:**
1. Client sends `POST /chat` with the full message history array.
2. The server injects a system prompt (containing catalog data and rules) plus the conversation history into a single Gemini API call.
3. Gemini returns a JSON object with `reply`, `recommendations`, and `end_of_conversation`.
4. A verification layer strips any recommendations whose URLs don't exist in the catalog.
5. The validated response is returned to the client.

**Why FastAPI?** It provides automatic schema validation via Pydantic, built-in OpenAPI docs, and async support — all important for a timed, auto-evaluated submission.

**Why Gemini 1.5 Flash?** It has a 1M token context window (easily holds the 377-item catalog), fast inference (critical for the 30-second timeout), and strong instruction-following for JSON output.

---

## 2. Retrieval Setup — Catalog-in-Context Strategy

Rather than using a vector database (FAISS + sentence-transformers) for retrieval, the full catalog is injected directly into the Gemini system prompt as a compact text table. Each of the 377 assessments is serialized into a single pipe-delimited row:

```
INDEX | NAME | TEST_TYPES | DURATION | REMOTE | ADAPTIVE | JOB_LEVELS | URL
```

This "catalog-in-context" approach was chosen over semantic retrieval for three reasons:

- **Accuracy:** Gemini sees every assessment on every turn, so it can never miss a relevant item due to retrieval error.
- **Simplicity:** No embedding model warm-up, no FAISS index initialization, no retrieval latency — all critical given the 30-second timeout.
- **Correctness guarantee:** The verification layer cross-checks every returned URL against the catalog set, making hallucination impossible in the final output.

The catalog text is trimmed to ~60,000 characters (~15,000 tokens) to stay safely within Gemini's context window while leaving room for the system prompt, conversation history, and generation output. This covers all 377 items comfortably.

**When would vector search help?** If the catalog grew beyond ~2,000 items, switching to FAISS pre-retrieval (top-50 candidates → inject into prompt) would reduce prompt size while maintaining recall.

---

## 3. Prompt Design

The system prompt has four sections:

**3.1 Strict Rules Block** — written in ALL-CAPS headers to make rule salience high. Covers: catalog-only constraint, clarify-before-recommend, 1–10 recommendation count, turn cap enforcement, off-topic refusal, and constraint refinement. Rules are numbered to make Gemini treat them as a checklist.

**3.2 Output Format Block** — specifies the exact JSON schema the model must emit. The instruction "respond with ONLY a valid JSON object, no markdown, no code fences" is reinforced by the extraction step in code (`extract_json()`) which strips any accidental fences.

**3.3 Catalog Block** — the pipe-delimited table of all assessments.

**3.4 Few-Shot Examples Block** — three short examples demonstrate:
- Vague query → clarifying question + empty recommendations
- Specific query → recommendations with catalog names/URLs
- Off-topic query → polite refusal + empty recommendations

**Temperature = 0.2** is used to make JSON output deterministic and reduce hallucination risk while preserving enough variability for natural language replies.

---

## 4. Behavioral Logic

| Behavior | Implementation |
|---|---|
| Clarify vague queries | System prompt rule + few-shot example; model returns `recommendations: []` |
| Turn cap (8 turns) | Server counts user turns; injects SYSTEM NOTE at turn 7; hard-stops at turn 8 |
| 30-second timeout | Measured in `call_gemini()`, raises 504 if exceeded |
| Off-topic refusal | System prompt rule + example; model refuses and returns empty recs |
| Constraint refinement | Full history re-sent each turn; model naturally updates recommendations |
| Catalog-only URLs | `verify_recommendations()` cross-checks every URL against catalog set |

---

## 5. Evaluation Approach

**Hard evals (pass/fail):**
- Schema compliance: Pydantic `ChatResponse` model enforces field presence and types at the API layer — malformed Gemini output triggers a safe fallback.
- Catalog-only URLs: `verify_recommendations()` drops any entry not in `CATALOG_URLS` set (O(1) lookup).
- Turn cap: server-side counter, not model-dependent.

**Recall@10:**
The model is prompted to return up to 10 assessments ranked by relevance. Because the full catalog is visible in context, Gemini can perform semantic matching (e.g., "Java developer" → Java 8, Core Java, OOP tests) without needing a separate retrieval step. The few-shot examples demonstrate that skill-based queries should match test_types like "Knowledge & Skills" with relevant names.

**Behavior probes:**
- Vague query probe: first turn with "I need an assessment" should return `recommendations: []` — enforced by rule #2 and the example.
- Off-topic probe: non-assessment queries return refusal — enforced by rule #5.
- Edit probe: changing constraints mid-conversation is handled naturally because the full history is re-sent and Gemini re-evaluates from scratch.

---

## 6. Deployment

Deployed on **Render.com** (free tier) as a Python web service. The `Procfile` starts uvicorn binding to `$PORT` (Render's dynamic port injection). The `GOOGLE_API_KEY` is stored as a Render environment variable — never in code or Git. The `data/catalog.json` file is committed to the repository so Render can access it at runtime.

**Estimated cold-start time:** ~15 seconds (Render free tier). Subsequent requests: 3–8 seconds (Gemini latency), well within the 30-second timeout.
