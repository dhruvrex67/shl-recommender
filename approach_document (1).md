# SHL Assessment Recommender — Approach Document

## Problem Decomposition

The core challenge is bridging vague hiring intent to a grounded shortlist from a 400+ item catalog. The main failure modes to guard against are: hallucinated assessments, recommendations on turn 1 for vague queries, ignoring refinements, and prompt injections.

I decomposed the problem into four layers:
1. **Catalog ingestion** — fetch, clean, and normalise the SHL JSON into a consistent schema
2. **Retrieval** — select the most relevant assessments per conversation turn
3. **Agent** — stateless LLM that decides when to clarify, recommend, refine, compare, or refuse
4. **API** — FastAPI service that validates schema and enforces turn caps

---

## Retrieval Design

**Approach:** TF-IDF cosine similarity with query expansion and semantic boosting.

The full catalog (~400 items) is too large to fit reliably in every prompt. Instead, I build a TF-IDF inverted index at startup (no external vector DB required, runs in <100ms per query). At each `/chat` call, I retrieve the top 25 most relevant assessments based on the accumulated user messages.

**Query expansion** addresses the vocabulary mismatch problem: "stakeholder management" doesn't appear in OPQ32r's description, but "interpersonal", "communication", and "behaviour" do. I maintain a synonym map (e.g. `developer → programmer, engineer, software`) applied before tokenisation.

**Semantic boosting:** if people-related keywords appear in the query (stakeholder, leadership, customer, sales, etc.), I force-include personality/OPQ-type assessments even if they score low on keyword overlap. This directly improves Recall@10 for mixed technical+behavioural roles.

**Name weighting:** the catalog entry name is repeated 3× and test types 2× in the document representation, so exact-match names score significantly higher than description-only matches.

---

## Prompt Design

The system prompt is injected fresh on every call with the dynamically retrieved catalog subset. Key design decisions:

- **Structured rules block** with clear section headers prevents rule amnesia over long conversations
- **Explicit test_type codes** (K, P, S, A, B, C, D, E) match the evaluator's expected format
- **Few-shot examples** for all four behaviors: clarify, recommend, refuse, refine
- **Turn cap reminder** appended as a user-message suffix on turn 7, forcing a final recommendation
- **Temperature 0.1** minimises hallucination while allowing natural phrasing

---

## Agent Design

The agent is stateless: every `/chat` call receives the full conversation history and the retrieved catalog. No per-session state is stored.

**Clarify:** on vague queries (no role mentioned), the LLM is instructed to ask exactly one question. The few-shot examples anchor this behavior.

**Recommend:** once role + one qualifier (seniority/skills/type) are present, recommend 1–10 catalog items with codes.

**Refine:** the full conversation history means prior recommendations are visible in context; the LLM updates rather than restarts.

**Compare:** uses only catalog fields (type, duration, remote, adaptive, levels, description) — no prior model knowledge.

**Refuse:** off-topic and injection attempts are handled at two layers: (1) a regex pre-check before the LLM call, (2) explicit instructions in the system prompt with an exact response template.

---

## JSON Robustness

LLMs occasionally produce malformed JSON. I implement four parsing strategies in sequence:
1. Direct `json.loads`
2. Strip trailing commas (`re.sub`)
3. Replace unescaped newlines inside strings
4. Regex extraction of the `reply` field as a minimum fallback

The fallback response is a natural clarifying question, not an error message, so the conversation stays coherent.

---

## Evaluation

**Hard evals:** schema compliance validated manually and via curl. Turn cap tested by sending 9-turn history. URL verification ensures every recommended URL exists in `CATALOG_URLS`.

**Recall@10:** tested against public traces. Key improvement: adding query expansion lifted Java-developer persona from 3/5 expected items to 5/5 by including OPQ32r and Automata tests.

**Behavior probes tested:**
- Vague query → clarifies ✅
- Specific query → recommends ✅
- Off-topic → refuses ✅
- Prompt injection → refuses ✅
- Refinement ("add personality") → updates shortlist ✅
- Comparison → grounded in catalog data ✅
- Turn 7 → forces end_of_conversation=true ✅

---

## Stack

- **LLM:** Groq (llama-3.3-70b-versatile) — free tier, ~1-2s latency, fits 27s timeout
- **Retrieval:** custom TF-IDF (no FAISS/Chroma dependency, cold-start friendly)
- **Framework:** FastAPI + Pydantic v2
- **Deployment:** Render (free tier, Procfile-based)
- **AI tools used:** Claude (Anthropic) for code review and prompt iteration

**What didn't work:**
- Putting the full catalog in every prompt: too slow (8-10s) and caused the LLM to ignore low-ranked items
- Using Groq's default temperature (0.7): produced inconsistent JSON structure
- Single-letter test_type in the initial prompt without the code legend: LLM used wrong codes
