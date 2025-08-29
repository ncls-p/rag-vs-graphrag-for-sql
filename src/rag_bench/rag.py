from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json as _json
import re as _re

from .config import Config
from .llm import OpenAIChat
from .retrievals.neo4j import Neo4jRetriever
from .retrievals.qdrant import QdrantRetriever


@dataclass
class RagAnswer:
    backend: str
    question: str
    retrieval_query: Optional[str]
    answer: str
    contexts: List[Dict[str, Any]]
    model: str
    usage: Dict[str, Any]


def _format_contexts(hits, max_chars: int = 6000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = 0
    for h in hits:
        payload = h.payload or {}
        content = payload.get("content") or ""
        if not content:
            continue
        content = str(content)
        if total + len(content) > max_chars:
            content = content[: max(0, max_chars - total)]
        out.append({"id": h.id, "score": h.score, "content": content})
        total += len(content)
        if total >= max_chars:
            break
    return out


def generate_search_query(
    question: str,
    backend: str = "qdrant",
    source_format: Optional[str] = None,
    instruction: Optional[str] = None,
) -> str:
    """Use the LLM to generate a compact retrieval query from the user's question.
    Returns a single-line query string to feed the retriever (embedding-based).
    """
    sys_prompt = (
        " - Your role: answer questions about tables, columns, SQL querys and values accurately by consulting the most relevant files here."
        " - The user’s questions are about DRLZ1 structure and data semantics. Select files by decoding their names and confirm details in the file contents."
        " ## How to find relevant file"
        " - Query the DRLZ1 tool to get informations about the database"
        " - Decode the filename into tokens (prefix, domain, subject, granularity, modifiers)."
        " - Filter candidates via those tokens (e.g., domain=CICS, subject=TRAN, granularity=D for daily)."
        " - Open the file(s) and verify columns/description match the question."
        " - Names ending with V are typically view-like aggregations; however, this export may still define them as TABLE in the SQL DDL. Use the name for intent, and the DDL to confirm the actual object type."
        " ## Filename anatomy"
        " Format (typical):"
        " [prefix]_[domain]_[subject]_[granularity][type/modifiers]"
        " - Prefix (purpose)"
        "   - CP = Capacity Planning artifacts (derived/aggregated, planning-oriented)"
        "   - No CP (e.g., CICS..., CICSWEB...) = Base/lookup/domain-specific artifacts"
        " - Domain (system/area)"
        "   - CICS = Customer Information Control System"
        "   - DB2 = Db2 correlation (CICS+Db2 join)"
        "   - CEC = Central Electronic Complex (hardware frame)"
        "   - CPU = CPU-related metrics"
        "   - CHANNEL = I/O channel metrics (SMF 73)"
        "   - WEB = CICS Web Interface"
        "   - BTS = CICS Business Transaction Services"
        "   - CHN = CICS Channels and Containers"
        "   - DOC = CICS document processing"
        "   - TIMEZONES, LOOKUP = Lookup/reference tables"
        " - Subject / scope"
        "   - TRAN = Transaction-level"
        "   - USR = Per-user grouping (often with TRAN as TRAN_USR)"
        "   - APPL = Application grouping (mapped via lookups)"
        "   - REGN = CICS Region-level"
        "   - SYS = System-level"
        "   - WKLD = Workload-level"
        "   - MIPS = MIPS/MSU capacity measures"
        "   - LPAR = Logical partition-level"
        "   - PROF = Profile-style aggregates"
        " - Granularity letters"
        "   - T = Timestamp-level"
        "   - H = Hourly"
        "   - D = Daily"
        "   - W = Weekly"
        "   - M = Monthly"
        " - Object type / modifiers"
        "   - V = View-like (DV, HV, MV, TV, WV) – confirm via SQL DDL."
        "   - F = Forecast (fields like F_ALGORITHM, F_TIMESTAMP, F_..., or FV__)."
        "   - C = Processor-type/CPU dimension (PROCESSOR_TYPE present)."
        "   - M variant (e.g., TRANM, APPLM, REGNM) = MIPS-focused."
        "   - Numeric variant (e.g., HV2) = second variant."
        " ## Decoding examples"
        " - CICS_DB2_TRANS_DV"
        "   - Domain: CICS + DB2; Subject: transactions; Granularity: Daily; Type: View"
        " - CP_CICS_TRAN_TV"
        "   - Capacity Planning; CICS transactions; Timestamp; View"
        " - CP_CICS_TRANC_T"
        "   - Capacity Planning; CICS transactions with processor-type dimension (C); Timestamp; Table"
        " - CP_CICS_TRANM_DV"
        "   - Capacity Planning; CICS transactions (MIPS variant); Daily; View"
        " - CP_CICS_APPL_T / CP_CICS_APPLC_T / CP_CICS_APPLM_TV"
        "   - APPL: by application; C (APPLC) adds processor-type dimension; M (APPLM) = MIPS variant; T/TV = timestamp/table-or-view"
        " - CP_CICS_REGNC_T"
        "   - Region-level with processor-type dimension; Timestamp; Table"
        " - CICSWEB_TRAN_USR_W"
        "   - CICS Web Interface; Transactions grouped by user; Weekly; Table"
        " - CP_CEC_WKLD_T"
        "   - CEC workload metrics at Timestamp granularity (capacity planning)"
        " - CP_CEC_MIPS_T"
        "   - CEC MIPS/MSU capacity at Timestamp granularity"
        " - CP_CHANNEL_F"
        "   - Channel metrics Forecast"
        " - CP_CEC_PROF_HV2"
        "   - CEC profile metrics; Hourly View; variant 2"
        " - CICS_TIMEZONES"
        "   - Lookup mapping MVS system to GMT offset (minutes)"
        " - CP_BUSINESS_APPL"
        "   - Lookup mapping (MVS system, WLM workload/service class/period) to Business application name"
        " ## Common lookups and joins"
        " - CP_BUSINESS_APPL: maps WORKLOAD and SERVICE_CLASS to BUSINESS_APPL (by MVS_SYSTEM_ID and SERV_CLASS_PERIOD)"
        " - CICS_TIMEZONES: maps MVS_SYSTEM_ID -> GMT offset minutes"
        " - CP_SHIFT_NM: used by many tables to derive SHIFT from date/time"
        " ## Choosing granularity"
        " - Timestamp (T/TV): fine-grained analysis and time-series KPIs"
        " - H/D/W/M: hourly/daily/weekly/monthly rollups for summarization/trending"
        " - V vs (no V): prefer Views for reporting/aggregations; Tables for base/atomic or non-aggregated structures"
        " - F: use Forecast tables for predicted values (fields like F_* or FV_*)"
        " ## Practical search tips"
        " - Need CICS + Db2 correlation? Search for tokens CICS and DB2 together"
        " - Per-user transaction stats? Look for *TRANUSR*_ or USR in the subject"
        " - Application perspective? APPL / APPLC / APPLM"
        " - Region rollups? REGN / REGNC / REGNM"
        " - MIPS/Capacity views? *MIPS* or subject ending with M (TRANM/APPLM/REGNM)"
        " - Hardware/LPAR/CEC? CP_CEC_, CP_CPU_, LPAR_"
        " - Weekly/Monthly reports? Use _W or _M; Daily _D; Hourly _H; Timestamp _T"
        " ## Validation cues inside files"
        " - PROCESSOR_TYPE column confirms a “C” variant (APPLC, REGNC, TRANC)"
        " - Presence of MIPS/MSU fields confirms an “M” variant (APPLM/TRANM/REGNM)"
        " - Descriptions may say “view” while DDL shows TABLE. Treat trailing V as view-like semantics, but rely on the SQL DDL to confirm the actual object type."
        " Note: Some lookup tables referenced in column descriptions (e.g., CP_SHIFT_NM) may not be present in this folder subset, but are commonly used to derive fields like SHIFT."
        " ## Short workflow for answering questions"
        " 1. Parse the user question to identify domain, subject, grouping, and desired time granularity."
        " 2. Select candidate files by filename tokens (using the rules above)."
        " 3. Open the candidate files; verify via the in-file description and column list."
        " 4. Extract the needed column definitions or DDL to answer precisely."
        " 5. If multiple sources apply, explain differences (e.g., TRAN vs TRANC vs TRANM)."
        " Remember you should always check the database documents before answering any question."
        " If you want to write sql, write is using code block markdown (```sql```)"
    )
    if instruction:
        sys_prompt = instruction
    fmt = f" format={source_format}" if source_format else ""
    user = f"backend={backend}{fmt}\nquestion: {question}"
    chat = OpenAIChat()
    res = chat.chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
    )
    # Sanitize the LLM output to a single plain line of tokens
    content = (res.content or "").strip()

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # remove leading fence
            s = s[3:]
            # drop language tag if present
            s = s.lstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")
            # cut at ending fence if any
            end = s.find("```")
            if end != -1:
                s = s[:end]
        return s

    def _from_json_like(s: str) -> Optional[str]:
        s = s.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                obj = _json.loads(s)
                # common shapes: {"query":"..."} or {"keywords":[...]} or list of strings
                if isinstance(obj, dict):
                    if isinstance(obj.get("query"), str):
                        return obj["query"].strip()
                    kw = obj.get("keywords")
                    if isinstance(kw, list):
                        return " ".join(str(x) for x in kw if x)
                if isinstance(obj, list):
                    return " ".join(str(x) for x in obj if isinstance(x, str))
            except Exception:
                return None
        return None

    raw = _strip_code_fences(content)
    extracted = _from_json_like(raw)
    if extracted is not None:
        raw = extracted
    # Collapse to single line and keep allowed characters
    raw = raw.replace("\n", " ").replace("\r", " ")
    # Allow alphanum, underscore, dash, slash, dot and space; drop the rest
    raw = _re.sub(r"[^A-Za-z0-9_./\- ]+", " ", raw)
    # Compress spaces
    raw = _re.sub(r"\s+", " ", raw).strip()
    # If too short or obviously broken (e.g., just a brace), fallback to question
    if len(raw) < 3 or raw in {"{", "[", ")", "("}:
        return question
    return raw


def answer_question(
    question: str,
    backend: str = "qdrant",
    top_k: Optional[int] = None,
    source_format: Optional[str] = None,
    source_formats: Optional[List[str]] = None,
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    retrieval_query: Optional[str] = None,
) -> RagAnswer:
    cfg = Config()
    if backend not in {"qdrant", "neo4j"}:
        raise ValueError("backend must be 'qdrant' or 'neo4j'")

    if backend == "qdrant":
        retr = QdrantRetriever(cfg)
    else:
        retr = Neo4jRetriever(cfg)
    try:
        rq = retrieval_query or question
        hits = retr.search(
            query=rq,
            top_k=top_k,
            source_format=source_format,
            source_formats=source_formats,
        )
    finally:
        closer = getattr(retr, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    contexts = _format_contexts(hits)
    ctx_text = "\n\n".join(f"[doc:{c['id']}]\n{c['content']}" for c in contexts)

    system = "You are a helpful assistant. Answer strictly using the provided context. If the answer is not in the context, say you don't know."
    user = f"Question:\n{question}\n\nRetrieval query:\n{retrieval_query or question}\n\nContext:\n{ctx_text}"

    chat = OpenAIChat()
    res = chat.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    usage = {
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        "elapsed_ms": res.elapsed_ms,
    }
    return RagAnswer(
        backend=backend,
        question=question,
        retrieval_query=retrieval_query or rq,
        answer=res.content,
        contexts=contexts,
        model=res.model,
        usage=usage,
    )
