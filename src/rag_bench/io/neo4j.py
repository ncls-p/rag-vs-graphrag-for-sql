from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from ..config import Config
from ..dataloader import load_records
from ..embeddings import OllamaEmbedder
from ..utils.text import combined_text


def _classify_entity(name: str) -> str:
    if "." in name:
        return "COLUMN"
    if "_" in name:
        return "TABLE"
    if name.isupper():
        return "TABLE"
    return "UNKNOWN"


@dataclass
class IngestStats:
    documents: int
    entities: int
    mentions: int
    refers_to: int
    skipped: int = 0


class Neo4jIO:
    def __init__(
        self,
        cfg: Optional[Config] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        use_read_only: bool = False,
    ) -> None:
        self.cfg = cfg or Config()
        self.uri = uri or self.cfg.neo4j_uri

        if use_read_only:
            self.user = user or self.cfg.neo4j_read_only_user
            self.password = password or self.cfg.neo4j_read_only_password
        else:
            self.user = user or self.cfg.neo4j_user
            self.password = password or self.cfg.neo4j_password

        if not self.user or not self.password:
            raise ValueError(
                "Neo4j credentials not provided. Set NEO4J_USER/NEO4J_PASSWORD (and RO variants for read-only)."
            )

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    # Admin / constraints
    def ensure_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        with self.driver.session() as s:
            for cql in stmts:
                s.execute_write(lambda tx, q: tx.run(q), cql)

    def drop_graph(self) -> Dict[str, Any]:
        if not self.cfg.allow_destructive_ops:
            return {
                "deleted_nodes": 0,
                "deleted_rels": 0,
                "note": "destructive ops disabled",
            }

        with self.driver.session() as s:
            res1 = s.execute_write(
                lambda tx: tx.run(
                    "MATCH ()-[r]-() DELETE r RETURN count(r) AS c"
                ).single()
            )
            rels = res1["c"] if res1 and "c" in res1 else 0
            res2 = s.execute_write(
                lambda tx: tx.run("MATCH (n) DELETE n RETURN count(n) AS c").single()
            )
            nodes = res2["c"] if res2 and "c" in res2 else 0
            return {"deleted_nodes": nodes, "deleted_rels": rels}

    # Ingestion
    @staticmethod
    def _merge_document_tx(tx, rec: Dict[str, Any], embedding: List[float]) -> None:
        sf = (rec.get("source_format") or "unknown").lower()
        tx.run(
            """
            MERGE (d:Document {id: $id})
            SET d.question = $question,
                d.answer_text = $answer_text,
                d.entities = $entities,
                d.doc_type = $doc_type,
                d.source_format = $source_format,
                d.embedding = $embedding
            FOREACH (_ IN CASE WHEN $sf = 'json' THEN [1] ELSE [] END | SET d:JSON)
            FOREACH (_ IN CASE WHEN $sf = 'txt' THEN [1] ELSE [] END | SET d:TXT)
            FOREACH (_ IN CASE WHEN $sf = 'xml' THEN [1] ELSE [] END | SET d:XML)
            """,
            id=rec["id"],
            question=rec.get("question", ""),
            answer_text=rec.get("answer_text", ""),
            entities=rec.get("entities", []),
            doc_type=rec.get("doc_type", "PLAIN"),
            source_format=rec.get("source_format", "unknown"),
            sf=sf,
            embedding=embedding,
        )

    @staticmethod
    def _merge_entity_and_rel_tx(tx, doc_id: int, ent_name: str, kind: str) -> None:
        tx.run(
            """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.kind = $kind
            WITH e
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            name=ent_name,
            kind=kind,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_refers_to_tx(tx, a: int, b: int, overlap: int, jaccard: float) -> None:
        tx.run(
            """
            MERGE (da:Document {id: $a})
            MERGE (db:Document {id: $b})
            MERGE (da)-[r:REFERS_TO]->(db)
            SET r.overlap_count = $overlap,
                r.jaccard = $jaccard
            """,
            a=a,
            b=b,
            overlap=overlap,
            jaccard=jaccard,
        )

    def ingest(
        self,
        data_path: Path,
        batch_size: int = 32,
        create_refers_to: bool = True,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> IngestStats:
        records = load_records(data_path)
        if not records:
            return IngestStats(documents=0, entities=0, mentions=0, refers_to=0)

        self.ensure_constraints()

        embedder = OllamaEmbedder()

        entity_to_docs: Dict[str, Set[int]] = defaultdict(set)
        doc_to_entities: Dict[int, Set[str]] = {}

        doc_count = 0
        ent_count = 0
        mention_count = 0
        skipped_count = 0

        # Progress metadata
        total_records = len(records)
        files_order: List[str] = []
        seen_files: Set[str] = set()
        for rec in records:
            sp = str(rec.get("source_path") or "")
            if sp and sp not in seen_files:
                seen_files.add(sp)
                files_order.append(sp)
        total_files = len(files_order) if files_order else 1
        file_index = 0
        current_file: Optional[str] = None
        processed = 0
        if progress:
            try:
                progress(
                    {
                        "backend": "neo4j",
                        "total_files": total_files,
                        "total_records": total_records,
                        "file_index": 0,
                        "file_path": None,
                        "record_index": 0,
                    }
                )
            except Exception:
                pass

        with self.driver.session() as s:
            for i, rec in enumerate(records, start=1):
                # Progress
                spath = str(rec.get("source_path") or "")
                if spath and spath != current_file:
                    current_file = spath
                    file_index += 1
                processed += 1
                if progress:
                    try:
                        progress(
                            {
                                "backend": "neo4j",
                                "total_files": total_files,
                                "total_records": total_records,
                                "file_index": file_index,
                                "file_path": current_file,
                                "record_index": processed,
                            }
                        )
                    except Exception:
                        pass
                text = combined_text(rec)
                vec: Optional[List[float]] = None
                try:
                    vec = embedder.embed_one(text).vector
                except Exception:
                    try:
                        vec = embedder.embed_one(text).vector
                    except Exception:
                        skipped_count += 1
                        continue

                # Write document and entities with per-op safety
                try:
                    s.execute_write(self._merge_document_tx, rec, vec)
                except Exception:
                    skipped_count += 1
                    continue
                doc_count += 1

                ents: List[str] = list(rec.get("entities", []) or [])
                doc_to_entities[rec["id"]] = set(ents)
                for e in ents:
                    kind = _classify_entity(e)
                    try:
                        s.execute_write(
                            self._merge_entity_and_rel_tx, rec["id"], e, kind
                        )
                        ent_count += 1
                        mention_count += 1
                        entity_to_docs[e].add(rec["id"])
                    except Exception:
                        # skip this entity relation but continue others
                        continue

        refers_to_count = 0
        if create_refers_to:
            pairs_done: Set[Tuple[int, int]] = set()
            with self.driver.session() as s:
                for ent, doc_ids in entity_to_docs.items():
                    ids = list(doc_ids)
                    n = len(ids)
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            a = ids[i]
                            b = ids[j]
                            if (a, b) in pairs_done:
                                continue
                            A = doc_to_entities.get(a, set())
                            B = doc_to_entities.get(b, set())
                            if not A or not B:
                                continue
                            inter = A.intersection(B)
                            if not inter:
                                continue
                            union = A.union(B)
                            overlap = len(inter)
                            jacc = float(overlap) / float(len(union)) if union else 0.0
                            s.execute_write(
                                self._merge_refers_to_tx, a, b, overlap, jacc
                            )
                            refers_to_count += 1
                            pairs_done.add((a, b))

        return IngestStats(
            documents=doc_count,
            entities=ent_count,
            mentions=mention_count,
            refers_to=refers_to_count,
            skipped=skipped_count,
        )

    def stats(self) -> Dict[str, int]:
        with self.driver.session() as s:
            doc_res = s.execute_read(
                lambda tx: tx.run("MATCH (d:Document) RETURN count(d) AS c").single()
            )
            doc = doc_res["c"] if doc_res and "c" in doc_res else 0

            ent_res = s.execute_read(
                lambda tx: tx.run("MATCH (e:Entity) RETURN count(e) AS c").single()
            )
            ent = ent_res["c"] if ent_res and "c" in ent_res else 0

            men_res = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (:Document)-[r:MENTIONS]->(:Entity) RETURN count(r) AS c"
                ).single()
            )
            men = men_res["c"] if men_res and "c" in men_res else 0

            ref_res = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (:Document)-[r:REFERS_TO]->(:Document) RETURN count(r) AS c"
                ).single()
            )
            ref = ref_res["c"] if ref_res and "c" in ref_res else 0

            return {
                "documents": doc,
                "entities": ent,
                "mentions": men,
                "refers_to": ref,
            }
