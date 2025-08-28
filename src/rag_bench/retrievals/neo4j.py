from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import Config
from ..embeddings import OllamaEmbedder
from ..io.neo4j import Neo4jIO
from ..parser import extract_entities
from .scoring import cosine, jaccard
from .types import Hit


class Neo4jRetriever:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.embedder = OllamaEmbedder()
        self.n4j = Neo4jIO(self.cfg, use_read_only=True)

    def close(self) -> None:
        try:
            self.n4j.close()
        except Exception:
            pass

    def _load_docs(
        self, source_format: Optional[str] = None, source_formats: Optional[List[str]] = None
    ) -> Tuple[List[int], List[List[float]], List[Set[str]]]:
        ids: List[int] = []
        embs: List[List[float]] = []
        ents: List[Set[str]] = []
        with self.n4j.driver.session() as s:
            if source_formats:
                rows = s.execute_read(
                    lambda tx, sfs: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.source_format IN $sfs "
                            "RETURN d.id AS id, d.embedding AS embedding, d.entities AS entities",
                            sfs=[sf.lower() for sf in sfs],
                        )
                    ),
                    source_formats,
                )
            elif source_format:
                rows = s.execute_read(
                    lambda tx, sf: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.source_format = $sf "
                            "RETURN d.id AS id, d.embedding AS embedding, d.entities AS entities",
                            sf=sf,
                        )
                    ),
                    source_format.lower(),
                )
            else:
                rows = s.execute_read(
                    lambda tx: list(
                        tx.run(
                            "MATCH (d:Document) "
                            "RETURN d.id AS id, d.embedding AS embedding, d.entities AS entities"
                        )
                    )
                )
        for r in rows:
            ids.append(int(r["id"]))
            embs.append(list(r["embedding"] or []))
            ents.append(set((r["entities"] or [])))
        return ids, embs, ents

    def _load_neighbors(
        self, source_format: Optional[str] = None, source_formats: Optional[List[str]] = None
    ) -> Dict[int, List[Tuple[int, List[float]]]]:
        neighbors: Dict[int, List[Tuple[int, List[float]]]] = {}
        with self.n4j.driver.session() as s:
            if source_formats:
                rows = s.execute_read(
                    lambda tx, sfs: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.source_format IN $sfs "
                            "OPTIONAL MATCH (d)-[:REFERS_TO]->(n:Document) "
                            "WHERE n.source_format IN $sfs "
                            "RETURN d.id AS id, collect({id:n.id, embedding:n.embedding}) AS neighbors",
                            sfs=[sf.lower() for sf in sfs],
                        )
                    ),
                    source_formats,
                )
            elif source_format:
                rows = s.execute_read(
                    lambda tx, sf: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.source_format = $sf "
                            "OPTIONAL MATCH (d)-[:REFERS_TO]->(n:Document) "
                            "WHERE n.source_format = $sf "
                            "RETURN d.id AS id, collect({id:n.id, embedding:n.embedding}) AS neighbors",
                            sf=sf,
                        )
                    ),
                    source_format.lower(),
                )
            else:
                rows = s.execute_read(
                    lambda tx: list(
                        tx.run(
                            "MATCH (d:Document) "
                            "OPTIONAL MATCH (d)-[:REFERS_TO]->(n:Document) "
                            "RETURN d.id AS id, collect({id:n.id, embedding:n.embedding}) AS neighbors"
                        )
                    )
                )
        for r in rows:
            did = int(r["id"])
            neigh_list = []
            for n in r["neighbors"]:
                if n["id"] is None or n["embedding"] is None:
                    continue
                neigh_list.append((int(n["id"]), list(n["embedding"])))
            neighbors[did] = neigh_list
        return neighbors

    def _load_payloads(
        self, ids: List[int], source_format: Optional[str] = None, source_formats: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        payloads: Dict[int, Dict[str, Any]] = {}
        with self.n4j.driver.session() as s:
            if source_formats:
                rows = s.execute_read(
                    lambda tx, ids, sfs: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.id IN $ids AND d.source_format IN $sfs "
                            "RETURN d.id AS id, d.question AS question, d.answer_text AS answer_text, "
                            "d.entities AS entities, d.doc_type AS doc_type",
                            ids=ids,
                            sfs=[sf.lower() for sf in sfs],
                        )
                    ),
                    ids,
                    source_formats,
                )
            elif source_format:
                rows = s.execute_read(
                    lambda tx, ids, sf: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.id IN $ids AND d.source_format = $sf "
                            "RETURN d.id AS id, d.question AS question, d.answer_text AS answer_text, "
                            "d.entities AS entities, d.doc_type AS doc_type",
                            ids=ids,
                            sf=sf,
                        )
                    ),
                    ids,
                    source_format.lower(),
                )
            else:
                rows = s.execute_read(
                    lambda tx, ids: list(
                        tx.run(
                            "MATCH (d:Document) WHERE d.id IN $ids "
                            "RETURN d.id AS id, d.question AS question, d.answer_text AS answer_text, "
                            "d.entities AS entities, d.doc_type AS doc_type",
                            ids=ids,
                        )
                    ),
                    ids,
                )
        for r in rows:
            pid = int(r["id"])
            payloads[pid] = {
                "question": r["question"],
                "answer_text": r["answer_text"],
                "entities": r["entities"],
                "doc_type": r["doc_type"],
            }
        return payloads

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_format: Optional[str] = None,
        source_formats: Optional[List[str]] = None,
    ) -> List[Hit]:
        k = top_k or self.cfg.top_k
        qvec = self.embedder.embed_one(query).vector
        qents = set(extract_entities(query))

        ids, embs, ents = self._load_docs(source_format=source_format, source_formats=source_formats)
        if not ids:
            return []

        # Semantic shortlist
        sem_scores = [(cosine(qvec, v), i) for i, v in enumerate(embs)]
        sem_scores.sort(key=lambda x: x[0], reverse=True)

        shortlist_size = self.cfg.shortlist_size
        shortlist_idx: Set[int] = set(i for _, i in sem_scores[:shortlist_size])

        # Entity filter
        if qents:
            for i, e in enumerate(ents):
                if e.intersection(qents):
                    shortlist_idx.add(i)

        neighbors = self._load_neighbors(source_format=source_format, source_formats=source_formats)

        hits: List[Hit] = []
        for i in shortlist_idx:
            did = ids[i]
            dv = embs[i]
            dent = ents[i]

            semantic = cosine(qvec, dv)
            ent_jacc = jaccard(qents, dent)
            neighs = neighbors.get(did, [])
            neighbor_boost = 0.0
            for _, nv in neighs:
                c = cosine(qvec, nv)
                if c > neighbor_boost:
                    neighbor_boost = c

            score = (
                self.cfg.alpha * semantic
                + self.cfg.beta * ent_jacc
                + self.cfg.gamma * neighbor_boost
            )
            hits.append(
                Hit(
                    id=did,
                    score=score,
                    components={
                        "semantic": semantic,
                        "entity_jaccard": ent_jacc,
                        "neighbor_boost": neighbor_boost,
                    },
                )
            )

        # Rank and trim
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]

        # Attach payloads
        pay = self._load_payloads([h.id for h in hits], source_format=source_format, source_formats=source_formats)
        for h in hits:
            h.payload = pay.get(h.id)

        return hits
