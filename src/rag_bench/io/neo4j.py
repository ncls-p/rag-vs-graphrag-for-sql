from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from ..config import Config
from ..fileloader import load_files, FileItem
from ..embeddings import OllamaEmbedder


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
            # Keep document id uniqueness; no Entity constraints in file-based ingestion
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
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
        # Back-compat signature kept, but we expect a FileItem-like dict
        sf = (rec.get("source_format") or rec.get("sf") or "unknown").lower()
        tx.run(
            """
            MERGE (d:Document {id: $id})
            SET d.path = $path,
                d.source_format = $source_format,
                d.size_bytes = $size_bytes,
                d.embedding = $embedding
            FOREACH (_ IN CASE WHEN $sf = 'json' THEN [1] ELSE [] END | SET d:JSON)
            FOREACH (_ IN CASE WHEN $sf = 'txt' THEN [1] ELSE [] END | SET d:TXT)
            FOREACH (_ IN CASE WHEN $sf = 'xml' THEN [1] ELSE [] END | SET d:XML)
            """,
            id=rec["id"],
            path=rec.get("source_path") or rec.get("path", ""),
            source_format=rec.get("source_format", "unknown"),
            size_bytes=int(rec.get("size_bytes", 0)),
            sf=sf,
            embedding=embedding,
        )

    # No Entity graph in file-based ingestion

    @staticmethod
    def _merge_refers_to_tx(tx, a: int, b: int, reason: str) -> None:
        tx.run(
            """
            MERGE (da:Document {id: $a})
            MERGE (db:Document {id: $b})
            MERGE (da)-[r:REFERS_TO]->(db)
            SET r.reason = $reason
            """,
            a=a,
            b=b,
            reason=reason,
        )

    # --- Foreign key heuristics (DDL text) ---
    @staticmethod
    def _extract_table_defs(text: str) -> List[str]:
        import re
        names: List[str] = []
        for m in re.finditer(r"(?is)\bCREATE\s+TABLE\s+([`\"]?[A-Za-z_][\w$.]*[`\"]?)", text):
            raw = m.group(1)
            norm = raw.strip().strip('`"').upper()
            names.append(norm)
        return names

    @staticmethod
    def _extract_fk_refs(text: str) -> List[str]:
        import re
        refs: List[str] = []
        for m in re.finditer(r"(?is)\bREFERENCES\s+([`\"]?[A-Za-z_][\w$.]*[`\"]?)", text):
            raw = m.group(1)
            norm = raw.strip().strip('`"').upper()
            refs.append(norm)
        return refs

    def ingest(
        self,
        data_path: Path,
        batch_size: int = 32,
        create_refers_to: bool = True,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> IngestStats:
        files = load_files(data_path)
        if not files:
            return IngestStats(documents=0, entities=0, mentions=0, refers_to=0)

        self.ensure_constraints()

        embedder = OllamaEmbedder()

        # File-based graph pre-pass for FK edges
        tables_by_format: Dict[str, Dict[str, int]] = defaultdict(dict)  # sf -> table -> doc_id
        fkrels_by_file: Dict[int, List[str]] = defaultdict(list)
        doc_to_format: Dict[int, str] = {}

        doc_count = 0
        ent_count = 0
        mention_count = 0
        skipped_count = 0

        # Progress metadata
        total_records = len(files)
        files_order: List[str] = []
        seen_files: Set[str] = set()
        for f in files:
            sp = f.source_path
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
            for i, item in enumerate(files, start=1):
                # Progress
                spath = item.source_path
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
                text = item.content
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
                    # pass dict-like for back-compat
                    s.execute_write(
                        self._merge_document_tx,
                        {
                            "id": item.id,
                            "source_path": item.source_path,
                            "source_format": item.source_format,
                            "size_bytes": item.size_bytes,
                        },
                        vec,
                    )
                except Exception:
                    skipped_count += 1
                    continue
                doc_count += 1

                # Precompute FK graph data
                sf = (item.source_format or "unknown").lower()
                for t in self._extract_table_defs(text):
                    if t not in tables_by_format[sf]:
                        tables_by_format[sf][t] = item.id
                refs = self._extract_fk_refs(text)
                if refs:
                    fkrels_by_file[item.id].extend(refs)
                doc_to_format[item.id] = sf

        refers_to_count = 0
        if create_refers_to:
            with self.driver.session() as s:
                for sf, tmap in tables_by_format.items():
                    for src_doc, ref_tables in fkrels_by_file.items():
                        if doc_to_format.get(src_doc) != sf:
                            continue
                        for rt in ref_tables:
                            dst_doc = tmap.get(rt)
                            if not dst_doc:
                                continue
                            try:
                                s.execute_write(
                                    self._merge_refers_to_tx,
                                    src_doc,
                                    dst_doc,
                                    "FOREIGN_KEY",
                                )
                                refers_to_count += 1
                            except Exception:
                                continue

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

            ref_res = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (:Document)-[r:REFERS_TO]->(:Document) RETURN count(r) AS c"
                ).single()
            )
            ref = ref_res["c"] if ref_res and "c" in ref_res else 0

            return {
                "documents": doc,
                "entities": 0,
                "mentions": 0,
                "refers_to": ref,
            }
