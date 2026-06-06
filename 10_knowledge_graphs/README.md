# Week 10: Knowledge Graphs — Structured Reasoning Beyond RAG

RAG answers "what does the text *say* about X?" by semantic similarity. It cannot
reliably answer "**what city is the HQ of the company that acquired X?**" — that's a
*chain of relationships*, and a vector search has no notion of relationships, only
of which chunks *look* alike. This week builds the alternative: extract the facts as
explicit `(subject, predicate, object)` triples, store them as a directed graph, and
answer multi-hop questions by **traversing** it.

> **Mental model:** RAG retrieves *chunks by similarity*; a knowledge graph
> retrieves *facts by connection*. When the answer requires following 2–3 explicit
> hops, traversal beats similarity — and it's auditable: you can point at the exact
> path that produced the answer.

---

## Why this week exists

It's the capstone of the "give the LLM the right context" thread that runs from
Week 1 (paste text in) → Week 7 Lab 1 (RAG) → here. Each step retrieves better
context; this one changes the *shape* of the context from prose to structure.

| Capability RAG lacks | What the graph gives you | Where it lives |
|----------------------|--------------------------|----------------|
| Multi-hop reasoning | Follow a chain of predicates | `multi_hop_query()` |
| Relationship queries | Typed edges, not similar chunks | `find_by_predicate()`, `neighbors()` |
| Connecting two entities | Shortest path between nodes | `shortest_path()` |
| Auditability | The exact traversal path is the evidence | the `MultiDiGraph` itself |

---

## What you'll build

An agent that reads unstructured text, extracts entities and relationships as
typed triples (via structured LLM output), builds an in-memory directed graph with
NetworkX, answers multi-hop questions by traversing it, and persists the graph
to/from JSON — **no vector database required.**

## Learning objectives

- Extract `(subject, predicate, object)` triples from prose with **structured
  outputs** (Pydantic schema, not free-text parsing)
- Build and query a directed multigraph with NetworkX (neighbors, shortest path,
  predicate filters)
- Answer **multi-hop** questions: "Who runs the company that acquired X?"
- Persist a graph to/from JSON for reuse across sessions
- Decide **knowledge graph vs RAG vs hybrid** for a given query shape and justify it

---

## Labs

| Lab | Topic | Key pattern |
|-----|-------|-------------|
| `1_lab1.ipynb` | Triple extraction | LLM → typed `(subject, predicate, object)` triples with confidence |
| `2_lab2.ipynb` | Graph construction & query | NetworkX graph; neighbors, shortest path, predicate filter |
| `3_lab3.ipynb` | Multi-hop reasoning | Answer questions needing 2–3 graph hops via traversal |
| `4_lab4.ipynb` | Graph + RAG hybrid | Graph for structure, vector search for unstructured context |

## App

`app.py` — paste any text, watch the agent extract a knowledge graph, then ask
multi-hop questions about it.

---

## Key module: [`knowledge_graph.py`](knowledge_graph.py)

| Piece | Role |
|-------|------|
| `Triple` / `TripleList` | Pydantic models — the schema the LLM is *forced* to return |
| `extract_triples(text)` | `client.beta.chat.completions.parse(..., response_format=TripleList)` — structured extraction, no regex parsing |
| `KnowledgeGraph` | Wraps a NetworkX `MultiDiGraph`; predicates and confidence ride on the edges |
| `multi_hop_query(start, hops)` | Traverse a chain of predicates, e.g. `("Gwynne Shotwell", ["works_at", "located_in"])` |
| `shortest_path` / `neighbors` / `find_by_predicate` | The query surface over the graph |
| `answer_question(q)` | Serialise the graph back to triples and let the LLM answer **grounded only in the graph** |
| `to_dict` / `from_dict` / `save` / `load` | JSON round-trip for persistence |

### Why a `MultiDiGraph` (not a plain graph)?

- **Directed** because relationships have direction: `Musk --founded--> SpaceX` is
  not `SpaceX --founded--> Musk`. Direction is what makes `neighbors(direction=...)`
  and multi-hop traversal meaningful.
- **Multi** because two entities can have several distinct relationships at once
  (`Musk --founded--> Tesla` *and* `Musk --ceo_of--> Tesla`). A simple graph would
  silently collapse them.

### Extraction is structured-output discipline

`extract_triples` uses `response_format=TripleList`, so the model returns a typed
object — not a string you hope parses. Note `obj` instead of `object` (the latter
is a Python builtin), and a per-triple `confidence` so you can later filter weak
edges. This is the Week 1 "structured output" lesson applied to graph building.

---

## When to use a knowledge graph vs RAG

| Scenario | Use |
|----------|-----|
| "What city is the HQ of the company that acquired X?" | **Knowledge graph** (multi-hop) |
| "Who are the direct reports of the CEO?" | **Knowledge graph** (relationship) |
| "How are entity A and entity B connected?" | **Knowledge graph** (shortest path) |
| "What does this document say about pricing?" | **RAG** (semantic lookup) |
| "Summarise the main themes of this article" | **RAG** (unstructured synthesis) |
| "Find the policy clause most similar to this question" | **RAG** (similarity) |

Rule of thumb: **structured relationships and chains → graph; fuzzy semantic recall
→ RAG.** Lab 4's hybrid uses the graph for the skeleton and RAG to fill in the prose
the graph can't hold.

---

## Design decisions worth defending in an interview

- **Why graph over RAG for multi-hop?** Similarity search retrieves chunks that
  *resemble* the query; it has no mechanism to *follow a relationship*. To answer
  "the HQ of the company that acquired X" you must hop X → acquirer → HQ — three
  edges a vector index simply doesn't represent.
- **Why ground `answer_question` strictly in the graph?** The prompt says "If the
  answer cannot be found in the graph, say 'Not found in graph.'" — that bounds the
  LLM to the extracted facts and makes hallucination visible. The graph is the
  citation.
- **Why store `confidence` on edges?** Extraction is probabilistic. Keeping
  per-edge confidence lets downstream queries down-weight or drop shaky facts
  instead of treating every triple as ground truth.
- **Hybrid, not either/or.** The senior answer to "KG or RAG?" is usually "both,
  for different parts of the query" — exactly what Lab 4 builds.

---

## Gotchas

- **Extraction quality is the ceiling.** Garbage or inconsistent predicates (`works_at`
  vs `employed_by` vs `is_employee_of`) fracture the graph and break multi-hop
  traversal. The prompt enforces `underscore_case` verb phrases for exactly this
  reason — entity/predicate normalisation is the hard, unglamorous 80%.
- **`answer_question` serialises the whole graph into the prompt.** Fine for a
  demo; for a large graph you must retrieve a relevant *subgraph* first (e.g. an
  ego-graph around the question's entities) or you blow the context window and the
  cost.
- **Entity resolution is unsolved here.** "Elon Musk", "Musk", and "E. Musk" become
  three different nodes unless you canonicalise. Real systems add a resolution step.

## Dependencies

```bash
pip install networkx matplotlib
```
`openai`, `pydantic`, and `python-dotenv` come with the course environment.

## Setup

Uses the course `.env` and `OPENAI_API_KEY`. No external database — the graph lives
in memory and persists to a JSON file via `save()` / `load()`.

## Cost estimate

~$0.01–0.05 per document graphed (extraction is one structured call per text block;
Q&A is one short call). gpt-4o-mini is the default throughout.

---

## How this connects

- **← Week 1:** triple extraction *is* structured output; Q&A *is* context injection
  — now over a graph instead of a blob.
- **← Week 7 Lab 1:** RAG was "retrieve the right chunks"; this is "retrieve the
  right *facts and their connections*." Lab 4 fuses the two.
- **→ Capstone:** a production agent often needs both — a graph for entities and
  relationships, RAG for the prose around them. See [CAPSTONE.md](../CAPSTONE.md).

**Next:** the [Capstone](../CAPSTONE.md) — integrate every week into one defended
system.
