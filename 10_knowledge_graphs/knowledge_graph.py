"""
Knowledge Graph builder and query engine.
Uses NetworkX for graph storage and an LLM for triple extraction.
No external database required.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import BaseModel

load_dotenv(override=True)

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("networkx not installed — run: pip install networkx")

client = OpenAI()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Triple(BaseModel):
    subject: str
    predicate: str
    obj: str          # 'object' is a Python builtin, use 'obj'
    confidence: float = 1.0

class TripleList(BaseModel):
    triples: list[Triple]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract all factual relationships from the text as (subject, predicate, object) triples.

Rules:
- Subject and object should be specific named entities (people, organisations, places, products)
- Predicate should be a concise verb phrase ('works_at', 'acquired', 'founded', 'located_in')
- Use underscore_case for predicates
- Only include facts explicitly stated in the text
- Assign confidence 0.0–1.0 based on how clearly the text states the fact

Text:
{text}"""


def extract_triples(text: str, max_retries: int = 3) -> list[Triple]:
    """
    Extract knowledge graph triples from text using an LLM.

    Robustness: validates input, retries transient API errors with backoff, and
    returns [] rather than raising on an empty/garbage parse.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract structured knowledge from text. Be precise."},
                    {"role": "user", "content": EXTRACT_PROMPT.format(text=text)},
                ],
                response_format=TripleList,
            )
            parsed = response.choices[0].message.parsed
            return parsed.triples if parsed else []
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Triple extraction failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

def canonicalize_entity(name: str) -> str:
    """
    Normalise an entity surface form so 'Elon Musk', 'elon  musk', and
    'Elon Musk.' collapse to one canonical key.

    This is deliberately conservative (casing/whitespace/punctuation only). It
    will NOT merge 'Musk' with 'Elon Musk' — that needs the alias map below,
    because blindly merging on substrings is unsafe ('Apple' vs 'Apple Inc').
    """
    if not isinstance(name, str):
        return str(name)
    cleaned = re.sub(r"[.’']", "", name)        # drop trailing dots / apostrophes
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned.islower() or cleaned.isupper() else cleaned


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """In-memory directed knowledge graph backed by NetworkX."""

    def __init__(self, resolve_entities: bool = True):
        if not NX_AVAILABLE:
            raise ImportError("networkx required: pip install networkx")
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.triple_count = 0
        self.resolve_entities = resolve_entities
        # Optional manual aliases, e.g. {"Musk": "Elon Musk"}. Applied after
        # canonicalisation so you can merge surface forms canonicalisation can't.
        self.aliases: dict[str, str] = {}

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register that `alias` refers to the same entity as `canonical`."""
        self.aliases[self._norm(alias)] = canonical

    def _norm(self, name: str) -> str:
        return canonicalize_entity(name) if self.resolve_entities else name

    def _resolve(self, name: str) -> str:
        """Canonicalise then apply the alias map."""
        key = self._norm(name)
        return self.aliases.get(key, key)

    def add_triple(self, triple: Triple):
        self.graph.add_edge(
            self._resolve(triple.subject),
            self._resolve(triple.obj),
            predicate=triple.predicate,
            confidence=triple.confidence,
        )
        self.triple_count += 1

    def add_triples(self, triples: list[Triple]):
        for t in triples:
            self.add_triple(t)

    def add_text(self, text: str) -> list[Triple]:
        """Extract triples from text and add to graph."""
        triples = extract_triples(text)
        self.add_triples(triples)
        return triples

    def neighbors(self, entity: str, direction: str = "out") -> list[dict]:
        """Return entities connected to `entity`.
        direction='out': entities this entity points to
        direction='in':  entities that point to this entity
        direction='both': both

        The query entity is resolved through canonicalisation + aliases, so
        `neighbors('musk')` finds the same node as `neighbors('Elon Musk')`.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in', or 'both'")
        entity = self._resolve(entity)
        if entity not in self.graph:
            return []
        results = []
        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(entity, data=True):
                results.append({"entity": target, "predicate": data["predicate"], "direction": "→"})
        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(entity, data=True):
                results.append({"entity": source, "predicate": data["predicate"], "direction": "←"})
        return results

    def shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        """Find shortest path between two entities (both resolved first)."""
        try:
            return nx.shortest_path(self.graph, self._resolve(source), self._resolve(target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_by_predicate(self, predicate: str) -> list[tuple[str, str]]:
        """Return all (subject, object) pairs with the given predicate."""
        results = []
        for s, o, data in self.graph.edges(data=True):
            if data.get("predicate") == predicate:
                results.append((s, o))
        return results

    def multi_hop_query(self, start: str, hops: list[str]) -> list[str]:
        """
        Traverse the graph following a chain of predicates.
        Example: multi_hop_query('Alice', ['works_at', 'located_in'])
        → Find where Alice works, then find where that company is located.
        """
        current_entities = {self._resolve(start)}
        for predicate in hops:
            next_entities = set()
            for entity in current_entities:
                if entity not in self.graph:
                    continue
                for _, target, data in self.graph.out_edges(entity, data=True):
                    if data.get("predicate") == predicate:
                        next_entities.add(target)
            current_entities = next_entities
            if not current_entities:
                break
        return list(current_entities)

    def relevant_subgraph(self, entities: list[str], radius: int = 2) -> list[tuple[str, str, str]]:
        """
        Return the edges within `radius` hops of any of `entities` (in either
        direction). This is how `answer_question` stays scalable: instead of
        serialising the whole graph into the prompt, we serialise only the
        neighbourhood around the question's entities.
        """
        seeds = {self._resolve(e) for e in entities if self._resolve(e) in self.graph}
        if not seeds:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        keep: set[str] = set()
        for seed in seeds:
            # ego_graph = all nodes within `radius` hops of the seed
            keep |= set(nx.ego_graph(undirected, seed, radius=radius).nodes())
        return [
            (s, d["predicate"], o)
            for s, o, d in self.graph.edges(data=True)
            if s in keep and o in keep
        ]

    def _mentioned_entities(self, question: str) -> list[str]:
        """Cheap string match: which graph nodes are named in the question?"""
        q = question.lower()
        return [n for n in self.graph.nodes() if str(n).lower() in q]

    def answer_question(self, question: str, radius: int = 2, max_edges: int = 200) -> str:
        """
        Answer a question grounded ONLY in the graph.

        Scalability: if the question names entities in the graph, we send just
        their `radius`-hop subgraph; otherwise we fall back to the whole graph
        but cap the number of edges so we never blow the context window.
        """
        if not isinstance(question, str) or not question.strip():
            return "Not found in graph."

        mentioned = self._mentioned_entities(question)
        if mentioned:
            edges = self.relevant_subgraph(mentioned, radius=radius)
        else:
            edges = [(s, d["predicate"], o) for s, o, d in self.graph.edges(data=True)]

        if not edges:
            return "Not found in graph."
        edges = edges[:max_edges]
        triples_text = "\n".join(f"({s}) --[{p}]--> ({o})" for s, p, o in edges)

        prompt = f"""Answer the question using ONLY the knowledge graph provided.
If the answer cannot be found in the graph, say 'Not found in graph.'

Knowledge graph:
{triples_text}

Question: {question}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        edges = [
            {"source": s, "target": o, "predicate": d["predicate"], "confidence": d.get("confidence", 1.0)}
            for s, o, d in self.graph.edges(data=True)
        ]
        return {"nodes": list(self.graph.nodes()), "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Load from serialised dict."""
        kg = cls()
        for edge in data["edges"]:
            kg.add_triple(Triple(
                subject=edge["source"],
                predicate=edge["predicate"],
                obj=edge["target"],
                confidence=edge.get("confidence", 1.0),
            ))
        return kg

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "predicates": len({d["predicate"] for _, _, d in self.graph.edges(data=True)}),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEXT = """
    Elon Musk founded SpaceX in 2002 and Tesla in 2003. SpaceX is headquartered in Hawthorne, California.
    Tesla is headquartered in Austin, Texas. In 2022, Musk acquired Twitter and renamed it X.
    Gwynne Shotwell is the President of SpaceX and has worked there since 2002.
    Sam Altman is the CEO of OpenAI, which is based in San Francisco.
    Microsoft invested in OpenAI in 2019 and 2023.
    """

    print("Building knowledge graph from text...")
    kg = KnowledgeGraph()
    triples = kg.add_text(TEXT)

    print(f"\nExtracted {len(triples)} triples:")
    for t in triples:
        print(f"  ({t.subject}) --[{t.predicate}]--> ({t.obj})")

    print(f"\nGraph stats: {kg.stats()}")

    print("\nNeighbors of 'Elon Musk':")
    for n in kg.neighbors("Elon Musk"):
        print(f"  {n['direction']} [{n['predicate']}] {n['entity']}")

    print("\nMulti-hop: Where is the company SpaceX's president's employer headquartered?")
    result = kg.multi_hop_query("Gwynne Shotwell", ["works_at", "located_in"])
    print(f"  Answer: {result}")

    print("\nQ&A:")
    print(kg.answer_question("Who founded SpaceX and where is it located?"))
    print(kg.answer_question("What company did Microsoft invest in?"))
