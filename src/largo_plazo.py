# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import secrets
import shutil
import sqlite3
import threading
import unicodedata
import urllib.error
import urllib.request
import time
from pathlib import Path
from typing import Any

from config import (
    LTM_DIR,
    LTM_DB_PATH,
    MEM_ARCHIVE_BELOW,
    MEM_BEAM,
    MEM_DESCRIBE_BATCH,
    MEM_LIST_LIMIT,
    MEM_MAX_CHILDREN,
    MEM_MAX_TOKENS,
    MEM_MIN_AGE,
    MEM_NAV_MAX_CALLS,
    MEM_REQUEST_TIMEOUT_S,
    MEM_REVISIT_TICKS,
    MEM_TAU,
    MEM_THINKING,
    LTM_INGEST_ASSISTANT,
    LTM_INGEST_USER,
    LTM_MIN_CHARS,
    MEM_RECALL_EXPAND,
    RRF_K,
    LTM_INGEST_QUEUE_MAX,
)

LEVELS = ["planeta", "pais", "ciudad", "fabrica", "persona", "organo", "tejido"]
N = len(LEVELS)

LEVELS_TEXT = (
    "La memoria es un árbol de 8 niveles, de lo más general a lo más concreto: "
    "1 planeta (dominio más amplio), 2 pais (gran categoría), 3 ciudad (subcategoría), "
    "4 fabrica (grupo concreto), 5 persona (entidad o tema específico), 6 organo (aspecto de esa entidad), "
    "7 tejido (conjunto muy pequeño de datos relacionados), 8 celula (un único dato atómico)."
)

SYSTEM = (
    "Eres el gestor de memoria de un agente. " + LEVELS_TEXT +
    " Llamas 'el agente' al sistema y 'el usuario' a la persona que habla (su dueño). "
    "Responde SIEMPRE y SOLO con un JSON válido, sin texto adicional ni comentarios."
)

def slugify(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-")
    return s or "general"

def extract_json(text: str) -> Any:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).replace("```json", "```")
    dec = json.JSONDecoder(strict=False)
    for i, ch in enumerate(text):
        if ch not in ("{", "["):
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, (dict, list)):
                return obj
        except ValueError:
            continue
    raise ValueError("no hay JSON en la respuesta")

def clamp(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))

def parse_answer(raw: str) -> dict[str, Any]:
    """Parsea respuestas generadas para la suite de pruebas o auditoría."""
    raw = raw or ""
    if not raw or "No tengo ese dato en memoria" in raw or "aún se está conectando" in raw:
        return {"found": False, "answer": "", "sources": []}
    
    sources = []
    ans_lines = []
    in_sources = False
    for line in raw.splitlines():
        sline = line.strip()
        if not sline:
            continue
        if sline.startswith("📚 Trazabilidad") or sline.startswith("📚 Fuentes"):
            in_sources = True
            continue
        if in_sources:
            if sline.startswith("•"):
                sources.append(sline.lstrip("• ").strip())
            elif sline.startswith("(") and sline.endswith(")"):
                in_sources = False
        elif sline.startswith("(") and sline.endswith(")"):
            continue
        else:
            if sline.startswith("💬"):
                ans_lines.append(sline.lstrip("💬 ").strip())
            elif not in_sources:
                ans_lines.append(sline)
                
    ans = "\n".join(ans_lines).strip()
    return {"found": bool(ans), "answer": ans, "sources": sources}

class MemLLM:
    """Cliente HTTP estándar para llamadas a llama-server (GPU 0 / LTM)."""
    def __init__(self) -> None:
        import config as C
        self.url = C.LLM_LTM_URL.rstrip("/") + "/chat/completions"
        self.model = C.REMOTE_MODEL_ALIAS
        self.calls = 0

    def ask(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": MEM_MAX_TOKENS,
            "stream": False,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": MEM_THINKING},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=MEM_REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fallo HTTP {exc.code} del servidor LLM ({self.url}): {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Fallo de red/LLM ({self.url}): {type(exc).__name__}: {exc}") from exc

        self.calls += 1
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Respuesta LLM sin choices ({self.url}): {str(data)[:300]}")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(msg, dict):
            raise RuntimeError(f"Respuesta LLM sin message ({self.url}): {str(choices[0])[:300]}")
        text = (msg.get("content") or "").strip()
        if not text:
            text = (msg.get("reasoning_content") or "").strip()
        return text

    def ask_json(self, system: str, user: str, attempts: int = 2) -> Any:
        last: Exception | None = None
        for i in range(attempts):
            text = self.ask(system, user if i == 0 else user + "\n\nResponde SOLO con el JSON, sin texto adicional.")
            try:
                return extract_json(text)
            except ValueError as exc:
                last = exc
        raise ValueError(f"El modelo no devolvió JSON válido: {last}")

class MemoryTree:
    TASKS = ["repair", "decay", "describe", "dedupe", "selftest", "reevaluate", "split", "merge"]

    def __init__(self, root: Path, llm: MemLLM) -> None:
        self.root = Path(root).resolve()
        self.db_path = self.root / "memoria.db"  # [parche] respeta `root`
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.lock = threading.RLock()
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        
        with self.lock:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")
            
            def sql_eff(imp, updated_tick, cur_tick, tau):
                age = max(0, cur_tick - updated_tick)
                return imp * math.exp(-age / (tau * max(imp, 1.0) ** 3))
            self.conn.create_function("eff", 4, sql_eff)

        self._init_db()
        self._load_meta()
        self._ingest_q: queue.Queue = queue.Queue(maxsize=max(1, LTM_INGEST_QUEUE_MAX))
        self._ingest_thread: threading.Thread | None = None
        self._ingest_stop = threading.Event()
        self._ingest_guard = threading.Lock()

    def _init_db(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER,
                    level INTEGER NOT NULL,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    importance REAL DEFAULT 5.0,
                    celulas INTEGER DEFAULT 0,
                    dirty INTEGER DEFAULT 1,
                    created_tick INTEGER DEFAULT 0,
                    FOREIGN KEY(parent_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);")
            if "pinned" not in {r[1] for r in cur.execute("PRAGMA table_info(nodes)")}:
                cur.execute("ALTER TABLE nodes ADD COLUMN pinned INTEGER DEFAULT 0")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS cells (
                    id TEXT PRIMARY KEY,
                    node_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    kind TEXT DEFAULT 'dato',
                    fuente TEXT,
                    ingest_timestamp TEXT,
                    created_tick INTEGER NOT NULL,
                    updated_tick INTEGER NOT NULL,
                    last_access_tick INTEGER DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    ruta_original TEXT,
                    tested_rev INTEGER DEFAULT -1,
                    tested_tick INTEGER DEFAULT 0,
                    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cells_node ON cells(node_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cells_archived ON cells(archived);")

            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS cells_fts USING fts5(
                    cell_id UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                );
            """)

            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_ai AFTER INSERT ON cells BEGIN
                    INSERT INTO cells_fts(cell_id, text) VALUES (new.id, new.text);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_ad AFTER DELETE ON cells BEGIN
                    DELETE FROM cells_fts WHERE cell_id = old.id;
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_cells_au AFTER UPDATE OF text ON cells BEGIN
                    DELETE FROM cells_fts WHERE cell_id = old.id;
                    INSERT INTO cells_fts(cell_id, text) VALUES (new.id, new.text);
                END;
            """)
            self.conn.commit()

    def _load_meta(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT key, value FROM meta")
            rows = dict(cur.fetchall())
            self.tick = int(rows.get("tick", 0))
            self.step = int(rows.get("step", 0))
            self.rev = int(rows.get("rev", 0))
            try:
                self.stats = json.loads(rows.get("stats", "{}"))
            except Exception:
                self.stats = {}
            for k in ["selftest_ok", "selftest_fail", "selftest_recovered", "recall_exact",
                      "recall_lexico", "recall_semantica", "recall_embeddings", "recall_arbol", "recall_hibrido"]:
                self.stats.setdefault(k, 0)

    def _save_meta(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tick', ?)", (str(self.tick),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('step', ?)", (str(self.step),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('rev', ?)", (str(self.rev),))
            cur.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('stats', ?)", (json.dumps(self.stats),))
            self.conn.commit()

    def advance(self, n: int = 1) -> int:
        self.tick += max(1, int(n))
        self.step += 1
        self._save_meta()
        return self.tick

    def _bump(self) -> None:
        self.rev += 1
        self._save_meta()

    def _add_directive(self, text: str, importance: float = 9.0) -> None:
        """Compatibilidad con interfaz: registra una directiva conductual permanente."""
        self.store(text, importance=max(9.0, importance), kind="instruccion")

    def _path_str(self, node_id: int) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                WITH RECURSIVE path_cte AS (
                    SELECT id, parent_id, name, level, name as full_path
                    FROM nodes WHERE id = ?
                    UNION ALL
                    SELECT n.id, n.parent_id, n.name, n.level, n.name || ' › ' || p.full_path
                    FROM nodes n JOIN path_cte p ON n.id = p.parent_id
                )
                SELECT full_path FROM path_cte WHERE parent_id IS NULL;
            """, (node_id,))
            r = cur.fetchone()
            return r[0] if r else ""

    def _kids_sorted(self, parent_id: int | None) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.conn.cursor()
            if parent_id is None:
                cur.execute("SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY importance DESC, name ASC")
            else:
                cur.execute("SELECT * FROM nodes WHERE parent_id = ? ORDER BY importance DESC, name ASC", (parent_id,))
            return cur.fetchall()

    def _mark_dirty(self, node_id: int) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                WITH RECURSIVE ancestors AS (
                    SELECT id, parent_id FROM nodes WHERE id = ?
                    UNION ALL
                    SELECT n.id, n.parent_id FROM nodes n JOIN ancestors a ON n.id = a.parent_id
                )
                UPDATE nodes SET dirty = 1 WHERE id IN (SELECT id FROM ancestors);
            """, (node_id,))
            self.conn.commit()

    def _make_node(self, parent_id: int | None, spec: dict[str, Any], level: int) -> int:
        name = str(spec.get("name") or "general").strip()[:60] or "general"
        desc = str(spec.get("description") or "").strip()[:200]
        slug = slugify(name)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, description FROM nodes WHERE (parent_id IS ? OR parent_id = ?) AND slug = ?",
                (parent_id, parent_id, slug)
            )
            row = cur.fetchone()
            if row:
                if not row["description"] and desc:
                    cur.execute("UPDATE nodes SET description = ? WHERE id = ?", (desc, row["id"]))
                    self.conn.commit()
                return row["id"]
            cur.execute("""
                INSERT INTO nodes (parent_id, level, slug, name, description, dirty, created_tick)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (parent_id, level, slug, name, desc, self.tick))
            self.conn.commit()
            self._bump()
            return cur.lastrowid

    def eff(self, cell: dict[str, Any] | sqlite3.Row) -> float:
        imp = float(cell["importance"])
        age = max(0, self.tick - int(cell["updated_tick"]))
        return imp * math.exp(-age / (MEM_TAU * max(imp, 1.0) ** 3))

    def _reinforce(self, cell_id: str) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE cells 
                SET importance = MIN(10.0, importance + 1.0),
                    updated_tick = ?,
                    last_access_tick = ?,
                    access_count = access_count + 1
                WHERE id = ?
            """, (self.tick, self.tick, cell_id))
            self.conn.commit()

    def store(self, text: str, importance: Any, kind: str = "dato", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {"id": "", "path": "", "importance": 0.0, "duplicate": False, "empty": True}
        
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, node_id, importance FROM cells WHERE lower(text) = lower(?) AND archived = 0", (text,))
            exact = cur.fetchone()
            if exact:
                self._reinforce(exact["id"])
                return {
                    "id": exact["id"],
                    "path": self._path_str(exact["node_id"]),
                    "importance": min(10.0, exact["importance"] + 1.0),
                    "duplicate": True
                }

        imp = clamp(importance, 1, 10, 5)
        cur_node_id = None
        depth = 0

        while depth < N:
            kids = self._kids_sorted(cur_node_id)
            remaining = N - depth
            if kids:
                listing = "\n".join(f"- {k['slug']} ({k['name']}) — {k['description'] or 'sin descripción'}" for k in kids[:MEM_LIST_LIMIT])
                user = (
                    "Dato a memorizar: " + text + "\n"
                    "Nivel actual: " + LEVELS[depth] + " (quedan " + str(remaining) + " niveles de nodos hasta la célula).\n"
                    "Nodos existentes en este nivel:\n" + listing + "\n"
                    'Si alguno es el lugar adecuado devuelve {"use":"<slug exacto>","update_description":null}.\n'
                    'Si ninguno encaja devuelve {"use":null,"path":[{"name":"...","description":"..."}]} creando una rama nueva.'
                )
                try:
                    dec = self.llm.ask_json(SYSTEM, user)
                except Exception:
                    dec = {}
                if not isinstance(dec, dict):
                    dec = {}
                use_slug = dec.get("use")
                matched_kid = next((k for k in kids if k["slug"] == slugify(use_slug)), None) if use_slug else None
                if matched_kid:
                    cur_node_id = matched_kid["id"]
                    depth += 1
                    continue
                path_specs = dec.get("path")
                if not isinstance(path_specs, list) or not path_specs:
                    path_specs = [{"name": "general", "description": ""}] * remaining
            else:
                user = (
                    "Dato a memorizar: " + text + "\n"
                    "Crea la rama completa de " + str(remaining) + " nodos desde '" + LEVELS[depth] + "' hasta '" + LEVELS[-1] + "'.\n"
                    'Devuelve JSON: {"path":[{"name":"...","description":"..."}]}'
                )
                try:
                    obj = self.llm.ask_json(SYSTEM, user)
                    path_specs = obj.get("path") if isinstance(obj, dict) else []
                except Exception:
                    path_specs = []
                if not isinstance(path_specs, list) or not path_specs:
                    path_specs = [{"name": "general", "description": ""}] * remaining

            path_specs = list(path_specs[:remaining])
            while len(path_specs) < remaining:  # [parche] la célula siempre cuelga de una hoja
                path_specs.append({"name": "general", "description": ""})
            for s in path_specs:
                cur_node_id = self._make_node(cur_node_id, s if isinstance(s, dict) else {"name": str(s)}, depth)
                depth += 1
            break

        cid = f"{self.tick:07d}-{secrets.token_hex(6)}"
        with self.lock:
            cur = self.conn.cursor()
            fuente = meta.get("fuente") if meta else None
            ingest_ts = meta.get("ingest_timestamp") if meta else time.strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("""
                INSERT INTO cells (id, node_id, text, importance, kind, fuente, ingest_timestamp, created_tick, updated_tick)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cid, cur_node_id, text, imp, kind, fuente, ingest_ts, self.tick, self.tick))
            cur.execute("UPDATE nodes SET celulas = celulas + 1 WHERE id = ?", (cur_node_id,))
            self.conn.commit()

        self._mark_dirty(cur_node_id)
        self._bump()
        return {"id": cid, "path": self._path_str(cur_node_id), "importance": imp, "duplicate": False}

    # ---------- RECUPERACIÓN HÍBRIDA (exacta + FTS5/BM25 + expansión LLM, fusión RRF) ----------
    _STOP = frozenset((
        "como", "cual", "cuales", "cuando", "donde", "quien", "quienes", "que", "porque", "cuanto",
        "cuantos", "para", "por", "con", "sin", "sobre", "entre", "desde", "hasta", "del", "los", "las",
        "una", "uno", "unos", "unas", "este", "esta", "esto", "ese", "esa", "eso", "mis", "tus", "sus",
        "hay", "son", "fue", "era", "ser", "estar", "tiene", "tengo", "puedes", "puede", "dime",
        "dijiste", "dije", "the", "and", "what", "which", "who", "how", "you", "are", "was", "that",
        "this", "with", "from",
    ))

    @classmethod
    def _fts_terms(cls, text: Any, limit: int = 10) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for w in re.findall(r"[^\W_]+", str(text).lower(), flags=re.UNICODE):
            folded = "".join(ch for ch in unicodedata.normalize("NFKD", w) if not unicodedata.combining(ch))
            if len(folded) < 3 or folded in cls._STOP or folded in seen:
                continue
            seen.add(folded)
            out.append(w)
            if len(out) >= limit:
                break
        return out

    def _fts_search(self, terms: list[str], limit: int = 15) -> list[dict[str, Any]]:
        """BM25 sobre FTS5. Incluye células archivadas (van detrás de las vivas)."""
        if not terms:
            return []
        query = " OR ".join('"%s"*' % t.replace('"', "") for t in terms)
        with self.lock:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    "SELECT c.*, bm25(cells_fts) AS rank FROM cells_fts "
                    "JOIN cells c ON c.id = cells_fts.cell_id "
                    "WHERE cells_fts MATCH ? ORDER BY c.archived ASC, rank ASC LIMIT ?",
                    (query, limit),
                )
                return [dict(r) for r in cur.fetchall()]
            except sqlite3.Error:
                return []

    def _exact_search(self, question: str, limit: int = 10) -> list[dict[str, Any]]:
        pat = "%" + question.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM cells WHERE lower(text) LIKE ? ESCAPE '\\' AND archived = 0 LIMIT ?",
                        (pat, limit))
            return [dict(r) for r in cur.fetchall()]

    def _expand_query(self, question: str) -> list[str]:
        user = (
            "Pregunta del usuario: " + question + "\n"
            "Devuelve hasta 8 palabras clave, sinónimos o términos relacionados que podrían aparecer "
            "en un recuerdo que responda a la pregunta.\n"
            'Devuelve JSON: {"terminos": ["..."]}'
        )
        try:
            obj = self.llm.ask_json(SYSTEM, user)
        except Exception:
            return []
        terms = obj.get("terminos") if isinstance(obj, dict) else None
        return [str(t) for t in terms][:8] if isinstance(terms, list) else []

    def _restore(self, cell_id: str) -> bool:
        """Despierta una célula archivada que ha vuelto a ser consultada."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE cells SET archived = 0, ruta_original = NULL, importance = MAX(importance, 6.0), "
                "updated_tick = ?, last_access_tick = ?, access_count = access_count + 1 "
                "WHERE id = ? AND archived = 1",
                (self.tick, self.tick, cell_id),
            )
            changed = cur.rowcount > 0
            if changed:
                cur.execute(
                    "UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE cells.node_id = nodes.id "
                    "AND cells.archived = 0) WHERE id = (SELECT node_id FROM cells WHERE id = ?)",
                    (cell_id,),
                )
            self.conn.commit()
        if changed:
            self._bump()
        return changed

    def recall(self, question: str, reinforce: bool = True) -> dict[str, Any]:
        started_calls = self.llm.calls
        question = (question or "").strip()

        def empty(methods: list[str]) -> dict[str, Any]:
            return {"found": False, "answer": "", "cells": [], "sources": [], "path": "",
                    "calls": self.llm.calls - started_calls, "methods": methods}

        if not question:
            return empty([])

        exact_hits = self._exact_search(question)
        lexical_hits = self._fts_search(self._fts_terms(question))
        rankings: dict[str, list[dict[str, Any]]] = {
            "exacta": exact_hits,
            "léxica-FTS5 (BM25)": lexical_hits,
        }
        semantic_ids: set[str] = set()
        if MEM_RECALL_EXPAND and not exact_hits and not lexical_hits:
            expanded = self._expand_query(question)
            sem_hits = self._fts_search(self._fts_terms(" ".join(expanded))) if expanded else []
            if sem_hits:
                rankings["semántica (expansión LLM)"] = sem_hits
                semantic_ids = {c["id"] for c in sem_hits}

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for hits in rankings.values():
            for rank, c in enumerate(hits):
                scores[c["id"]] = scores.get(c["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
                by_id.setdefault(c["id"], c)
        if not scores:
            return empty([m for m, h in rankings.items() if h] or ["léxica-FTS5 (BM25)"])

        ordered = sorted(scores, key=lambda i: scores[i], reverse=True)[:18]
        candidates = [by_id[i] for i in ordered]
        methods = [m for m, h in rankings.items() if h]

        rows = []
        for c in candidates:
            tag = " [archivada]" if c.get("archived") else ""
            rows.append(f"[{c['id']}] [imp {float(c['importance']):.0f}] [ruta: {self._path_str(c['node_id'])}]{tag} {c['text']}")

        prompt = (
            "Responde a la pregunta usando exclusivamente los recuerdos proporcionados. "
            "Si ninguno responde a la pregunta, devuelve found=false.\n"
            "Pregunta: " + question + "\nRecuerdos encontrados:\n" + "\n".join(rows) + "\n"
            'Devuelve obligatoriamente este JSON: {"found": true, "answer": "respuesta clara y concisa", "used": ["id1"]} '
            'o {"found": false, "answer": "", "used": []}'
        )
        try:
            res_obj = self.llm.ask_json(SYSTEM, prompt)
        except Exception:
            res_obj = {}

        if not isinstance(res_obj, dict) or not res_obj.get("found"):
            return empty(methods)

        raw_used = res_obj.get("used", [])
        used_ids = [str(i) for i in raw_used] if isinstance(raw_used, list) else []
        used_cells = [by_id[i] for i in used_ids if i in by_id]
        used_ids = [c["id"] for c in used_cells]
        sources = sorted({c["fuente"] for c in used_cells if c.get("fuente")})
        paths = sorted({self._path_str(c["node_id"]) for c in used_cells})

        restored = 0
        if reinforce:
            for c in used_cells:
                if c.get("archived") and self._restore(c["id"]):
                    restored += 1
                self._reinforce(c["id"])
            ex_ids = {c["id"] for c in exact_hits}
            lx_ids = {c["id"] for c in lexical_hits}
            if any(i in ex_ids for i in used_ids):
                self.stats["recall_exact"] = self.stats.get("recall_exact", 0) + 1
            if any(i in lx_ids for i in used_ids):
                self.stats["recall_lexico"] = self.stats.get("recall_lexico", 0) + 1
            if any(i in semantic_ids for i in used_ids):
                self.stats["recall_semantica"] = self.stats.get("recall_semantica", 0) + 1
            self.stats["recall_hibrido"] = self.stats.get("recall_hibrido", 0) + 1
            self._save_meta()

        return {
            "found": True,
            "answer": str(res_obj.get("answer", "")),
            "cells": used_ids,
            "sources": sources,
            "path": " | ".join(paths),
            "calls": self.llm.calls - started_calls,
            "methods": methods,
            "restored": restored,
        }

    def process_user_message(self, message: str) -> str:
        """Compatibilidad con bucle.py en terminal interactiva."""
        msg = message.strip()
        if not msg:
            return ""
        if msg.endswith("?") or any(w in msg.lower() for w in ["qué", "que", "cómo", "como", "quién", "quien", "cuándo", "cuando", "dónde", "donde", "por qué", "cuál", "cual"]):
            res = self.recall(msg, reinforce=True)
            if res.get("found"):
                srcs = f"\n📚 Fuentes: {', '.join(res['sources'])}" if res.get("sources") else ""
                return f"💬 {res['answer']}{srcs} (Ruta: {res.get('path', '-')})"
            return "💬 No encontré información sobre eso en la memoria a largo plazo."
        else:
            r = self.store(msg, importance=7, kind="dato")
            tag = "reforzado" if r.get("duplicate") else "memorizado"
            return f"💾 [{tag}] Guardado en: {r.get('path', 'raíz')} (ID: {r.get('id', '-')})"

    # ---------- TAREAS DEL CICLO DE SUEÑO (HOMEOSTASIS COMPLETA) ----------
    def maintenance_step(self) -> tuple[str, str, bool]:
        name = self.TASKS[(self.step - 1) % len(self.TASKS)] if self.step > 0 else self.TASKS[0]
        calls0 = self.llm.calls
        worked = False
        try:
            report = getattr(self, "task_" + name)()
            worked = self.llm.calls != calls0 or "·" not in report
        except Exception as exc:
            report = f"error {type(exc).__name__}: {exc}"
        # [parche] sin self.advance(1): lo hace el bucle antes de llamar (si no, solo corrían 4 de 8 tareas)
        return name, report, worked

    def task_repair(self) -> str:
        """Recalcula contadores, poda ramas vacías (en cascada) y reconstruye FTS5 si se desincroniza."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells "
                "WHERE cells.node_id = nodes.id AND cells.archived = 0)"
            )
            # poda en cascada: se repite hasta que no quedan hojas vacías. Solo nodos SIN ninguna
            # célula (ni archivada: ON DELETE CASCADE la destruiría), sin hijos, no creados a mano
            # (pinned) y no creados en este mismo tick (store podría estar a punto de llenarlos).
            pruned = 0
            while True:
                cur.execute(
                    "DELETE FROM nodes WHERE pinned = 0 AND created_tick < ? "
                    "AND NOT EXISTS (SELECT 1 FROM cells WHERE cells.node_id = nodes.id) "
                    "AND NOT EXISTS (SELECT 1 FROM nodes ch WHERE ch.parent_id = nodes.id)",
                    (self.tick,),
                )
                if cur.rowcount <= 0:
                    break
                pruned += cur.rowcount
            cur.execute("SELECT COUNT(*) FROM cells")
            n_cells = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM cells_fts")
            n_fts = cur.fetchone()[0]
            rebuilt = False
            if n_cells != n_fts:
                cur.execute("DELETE FROM cells_fts")
                cur.execute("INSERT INTO cells_fts(cell_id, text) SELECT id, text FROM cells")
                rebuilt = True
            self.conn.commit()
        if pruned or rebuilt:
            self._bump()
            parts = []
            if pruned:
                parts.append(f"{pruned} nodo(s) vacío(s) podado(s)")
            if rebuilt:
                parts.append(f"índice FTS5 reconstruido ({n_fts} -> {n_cells})")
            return "; ".join(parts)
        return "· integridad B-Tree y FTS5 verificada"

    def task_decay(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE cells 
                SET archived = 1,
                    ruta_original = (SELECT name FROM nodes WHERE nodes.id = cells.node_id)
                WHERE archived = 0 
                  AND (? - updated_tick) >= ?
                  AND eff(importance, updated_tick, ?, ?) < ?
            """, (self.tick, MEM_MIN_AGE, self.tick, MEM_TAU, MEM_ARCHIVE_BELOW))
            count = cur.rowcount
            if count > 0:
                self.conn.commit()
                self._bump()
                return f"{count} célula(s) archivadas por baja importancia efectiva"
        return "· células vivas estables, ninguna que olvidar"

    def task_describe(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, level FROM nodes WHERE dirty = 1 ORDER BY level DESC LIMIT ?", (MEM_DESCRIBE_BATCH,))
            targets = [dict(r) for r in cur.fetchall()]

        if not targets:
            return "· descripciones al día"

        for n in targets:
            nid = n["id"]
            if n["level"] == N - 1:
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("SELECT text FROM cells WHERE node_id = ? AND archived = 0 LIMIT 10", (nid,))
                    items = [r["text"] for r in cur.fetchall()]
            else:
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("SELECT name, description FROM nodes WHERE parent_id = ? LIMIT 10", (nid,))
                    items = [f"{r['name']}: {r['description']}" for r in cur.fetchall()]
            
            if items:
                user = (
                    f"Nodo '{n['name']}' (nivel {LEVELS[n['level']]}).\n"
                    "Contenido:\n- " + "\n- ".join(items) + "\n"
                    "Escribe una descripción del ÁMBITO de este nodo en máx 20 palabras.\n"
                    'Devuelve: {"description": "..."}'
                )
                try:
                    obj = self.llm.ask_json(SYSTEM, user)
                    desc = str(obj.get("description", "")).strip()[:200]
                except Exception:
                    desc = ""
            else:
                desc = "Área general"

            with self.lock:
                cur = self.conn.cursor()
                cur.execute("UPDATE nodes SET description = ?, dirty = 0 WHERE id = ?", (desc, nid))
                self.conn.commit()

        return f"{len(targets)} descripciones sintetizadas"

    def task_dedupe(self) -> str:
        """Deduplicación activa: detecta hechos redundantes en un mismo tejido y los fusiona."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT node_id FROM cells 
                WHERE archived = 0 
                GROUP BY node_id 
                HAVING COUNT(*) >= 2 
                ORDER BY RANDOM() LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                return "· sin tejidos con múltiples células para deduplicar"
            node_id = row[0]
            cur.execute("SELECT id, text, importance FROM cells WHERE node_id = ? AND archived = 0", (node_id,))
            cells = [dict(r) for r in cur.fetchall()]

        for i in range(len(cells)):
            w1 = set(re.findall(r"\w+", cells[i]["text"].lower()))
            for j in range(i + 1, len(cells)):
                w2 = set(re.findall(r"\w+", cells[j]["text"].lower()))
                inter = len(w1 & w2)
                union = len(w1 | w2)
                sim = inter / union if union > 0 else 0.0
                
                if sim >= 0.55 or cells[i]["text"].strip().lower() == cells[j]["text"].strip().lower():
                    c1, c2 = cells[i], cells[j]
                    if c1["text"].strip().lower() == c2["text"].strip().lower():
                        merged_text = c1["text"]
                        new_imp = min(10.0, max(c1["importance"], c2["importance"]) + 0.5)
                    else:
                        prompt = (
                            f"Compara estos dos recuerdos del mismo tema:\n"
                            f"A: {c1['text']}\nB: {c2['text']}\n\n"
                            f"Si son redundantes o uno complementa al otro, unifícalos en una sola afirmación concisa. "
                            f'Devuelve JSON: {{"redundant": true, "merged": "...", "importance": 8}}.\n'
                            f'Si representan hechos distintos y deben conservarse ambos, devuelve: {{"redundant": false}}.'
                        )
                        try:
                            ans = self.llm.ask_json(SYSTEM, prompt)
                            if not ans.get("redundant"):
                                continue
                            merged_text = ans.get("merged", c1["text"])
                            new_imp = float(ans.get("importance", max(c1["importance"], c2["importance"])))
                        except Exception:
                            continue

                    with self.lock:
                        cur = self.conn.cursor()
                        cur.execute("UPDATE cells SET text = ?, importance = ?, updated_tick = ? WHERE id = ?",
                                    (merged_text, new_imp, self.tick, c1["id"]))
                        cur.execute("UPDATE cells SET archived = 1, updated_tick = ?, ruta_original = 'fusión con ' || ? WHERE id = ?",
                                    (self.tick, c1["id"], c2["id"]))
                        cur.execute("UPDATE nodes SET celulas = MAX(0, celulas - 1) WHERE id = ?", (node_id,))
                        self.conn.commit()
                    self._mark_dirty(node_id)
                    self._bump()
                    return f"fusión redundancia: [{c1['id']}] + [{c2['id']}] → «{merged_text[:35]}...»"

        return "· células revisadas, sin redundancias detectadas"

    def _mark_tested(self, cell_id: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE cells SET tested_rev = ?, tested_tick = ? WHERE id = ?",
                              (self.rev, self.tick, cell_id))
            self.conn.commit()

    def _reindex_cell(self, cell_id: str) -> bool:
        """Si la célula falta en el índice FTS5, la reinserta. True si hubo reparación."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cells_fts WHERE cell_id = ?", (cell_id,))
            if cur.fetchone()[0] > 0:
                return False
            cur.execute("INSERT INTO cells_fts(cell_id, text) SELECT id, text FROM cells WHERE id = ?", (cell_id,))
            self.conn.commit()
            return True

    def task_selftest(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM cells WHERE archived = 0 AND tested_rev < ? "
                        "ORDER BY tested_tick ASC LIMIT 1", (self.rev,))
            row = cur.fetchone()
        if not row:
            return "· todas las células verificadas en la revisión actual"
        cell = dict(row)

        user = f'Formula una pregunta con la que se buscaría este dato: "{cell["text"]}". Devuelve {{"pregunta": "..."}}'
        try:
            obj = self.llm.ask_json(SYSTEM, user)
        except Exception:
            return "· LLM no disponible para el autotest; se reintentará"
        q = obj.get("pregunta") if isinstance(obj, dict) else None
        if not q:
            self._mark_tested(cell["id"])  # evita quedarse atascado en esta célula
            return "· no se pudo formular una pregunta de prueba"

        res = self.recall(str(q), reinforce=False)
        self._mark_tested(cell["id"])
        if res["found"] and cell["id"] in res["cells"]:
            self.stats["selftest_ok"] = self.stats.get("selftest_ok", 0) + 1
            self._save_meta()
            return f"OK: «{q}»"

        self.stats["selftest_fail"] = self.stats.get("selftest_fail", 0) + 1
        repaired = self._reindex_cell(cell["id"])
        if repaired:
            self.stats["selftest_recovered"] = self.stats.get("selftest_recovered", 0) + 1
        self._save_meta()
        return f"FALLO: «{q}» no recuperó [{cell['id']}]" + (" - índice FTS5 reparado" if repaired else "")

    def task_reevaluate(self) -> str:
        """Reevaluación de importancia: refuerza memorias muy accedidas y rescata borderline."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, text, importance, access_count 
                FROM cells 
                WHERE archived = 0 AND access_count >= 3 AND importance < 9.0 
                ORDER BY access_count DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                cid = row["id"]
                new_imp = min(10.0, row["importance"] + 1.0)
                cur.execute("UPDATE cells SET importance = ?, access_count = 0, updated_tick = ? WHERE id = ?",
                            (new_imp, self.tick, cid))
                self.conn.commit()
                self._bump()
                return f"reforzada célula [{cid}] a imp {new_imp:.1f} por uso intensivo"

            cur.execute("""
                SELECT id, text, importance, updated_tick 
                FROM cells 
                WHERE archived = 0 AND (? - updated_tick) >= ? 
                  AND eff(importance, updated_tick, ?, ?) BETWEEN ? AND (? + 0.35)
                ORDER BY updated_tick ASC LIMIT 1
            """, (self.tick, MEM_MIN_AGE, self.tick, MEM_TAU, MEM_ARCHIVE_BELOW, MEM_ARCHIVE_BELOW))
            cand = cur.fetchone()

        if cand:
            prompt = (
                f"Evalúa si este conocimiento merece ser preservado de forma duradera:\n"
                f"«{cand['text']}»\n"
                f'Devuelve JSON: {{"preserve": true/false, "importance": <1 a 10>}}'
            )
            try:
                res = self.llm.ask_json(SYSTEM, prompt)
                if res.get("preserve"):
                    boosted = clamp(res.get("importance", 6.5), 5.0, 10.0, 6.5)
                    with self.lock:
                        cur = self.conn.cursor()
                        cur.execute("UPDATE cells SET importance = ?, updated_tick = ? WHERE id = ?",
                                    (boosted, self.tick, cand["id"]))
                        self.conn.commit()
                    self._bump()
                    return f"rescate homeostático: [{cand['id']}] reevaluada a imp {boosted:.1f}"
            except Exception:
                pass

        return "· importancias y prioridades en equilibrio"

    def task_split(self) -> str:
        """Partición activa: divide de forma autónoma tejidos saturados para mantener el árbol equilibrado."""
        with self.lock:
            cur = self.conn.cursor()
            # 1. Buscar tejidos saturados (más células que el límite MEM_MAX_CHILDREN)
            cur.execute("""
                SELECT id, name, parent_id, level, celulas 
                FROM nodes 
                WHERE level = ? AND celulas > ? 
                LIMIT 1
            """, (N - 1, MEM_MAX_CHILDREN))
            overloaded = cur.fetchone()

        if not overloaded:
            return f"· factor de ramificación óptimo (≤{MEM_MAX_CHILDREN} por nodo)"

        nid, nname, pid, lvl = overloaded["id"], overloaded["name"], overloaded["parent_id"], overloaded["level"]
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, text FROM cells WHERE node_id = ? AND archived = 0", (nid,))
            cells = [dict(r) for r in cur.fetchall()]

        if len(cells) < 4:
            return "· células insuficientes para partición"

        listing = "\n".join(f"- [{c['id']}] {c['text'][:80]}" for c in cells[:16])
        prompt = (
            f"El grupo de memoria '{nname}' tiene demasiados recuerdos ({len(cells)}).\n"
            f"Divídelos en 2 subgrupos temáticos distintos y coherentes:\n"
            f"{listing}\n\n"
            'Devuelve EXCLUSIVAMENTE este JSON: {"grupo_A": {"nombre": "...", "ids": ["id1"]}, "grupo_B": {"nombre": "...", "ids": ["id2"]}}'
        )
        try:
            res = self.llm.ask_json(SYSTEM, prompt)
            g_a = res.get("grupo_A", {})
            g_b = res.get("grupo_B", {})
            valid_ids = {c["id"] for c in cells}
            ids_b = [str(x) for x in g_b.get("ids", []) if str(x) in valid_ids]
            if len(ids_b) >= len(cells):
                ids_b = ids_b[: len(cells) // 2]

            if g_b.get("nombre") and ids_b:
                new_slug = slugify(g_b["nombre"])
                with self.lock:
                    cur = self.conn.cursor()
                    # Crear nuevo nodo hermano para albergar el grupo B
                    cur.execute("""
                        INSERT INTO nodes (parent_id, level, slug, name, description, dirty, created_tick)
                        VALUES (?, ?, ?, ?, ?, 1, ?)
                    """, (pid, lvl, new_slug, g_b["nombre"][:60], "Subdivisión balanceada", self.tick))
                    new_node_id = cur.lastrowid
                    
                    # Migrar células seleccionadas al nuevo nodo
                    placeholders = ",".join("?" for _ in ids_b)
                    cur.execute(f"UPDATE cells SET node_id = ? WHERE node_id = ? AND id IN ({placeholders})",
                                [new_node_id, nid] + ids_b)
                    
                    # Actualizar contadores en ambos nodos
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE node_id = ? AND archived = 0) WHERE id = ?", (nid, nid))
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE node_id = ? AND archived = 0) WHERE id = ?", (new_node_id, new_node_id))
                    self.conn.commit()

                self._mark_dirty(nid)
                self._mark_dirty(new_node_id)
                self._bump()
                return f"partición exitosa en '{nname}': creado hermano '{g_b['nombre']}' ({len(ids_b)} células migradas)"
        except Exception as e:
            return f"aviso en partición: {e}"

        return "· análisis de ramificación completado" 

    def task_merge(self) -> str:
        """Fusión de ramas huérfanas o casi vacías bajo un mismo padre."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT n1.id as id1, n2.id as id2, n1.name as name1, n2.name as name2, n1.parent_id
                FROM nodes n1 
                JOIN nodes n2 ON n1.parent_id = n2.parent_id AND n1.id < n2.id
                WHERE n1.celulas <= 1 AND n2.celulas <= 1 AND n1.level = ?
                LIMIT 1
            """, (N - 1,))
            pair = cur.fetchone()

        if not pair:
            return "· ramas integradas y sin hojas dispersas"

        name1, name2 = pair["name1"], pair["name2"]
        id1, id2 = pair["id1"], pair["id2"]
        prompt = (
            f"¿Pertenecen estos dos subtemas a un mismo concepto integrable?\n"
            f"Tema 1: {name1}\nTema 2: {name2}\n"
            f'Devuelve JSON: {{"mergeable": true/false, "merged_name": "..."}}'
        )
        try:
            res = self.llm.ask_json(SYSTEM, prompt)
            if res.get("mergeable"):
                new_name = str(res.get("merged_name") or name1).strip()[:60]
                with self.lock:
                    cur = self.conn.cursor()
                    cur.execute("UPDATE cells SET node_id = ? WHERE node_id = ?", (id1, id2))
                    cur.execute("DELETE FROM nodes WHERE id = ?", (id2,))
                    cur.execute("UPDATE nodes SET name = ?, dirty = 1 WHERE id = ?", (new_name, id1))
                    cur.execute("UPDATE nodes SET celulas = (SELECT COUNT(*) FROM cells WHERE node_id = ? AND archived = 0) WHERE id = ?", (id1, id1))
                    self.conn.commit()
                self._mark_dirty(id1)
                self._bump()
                return f"fusión de ramas: '{name1}' + '{name2}' → '{new_name}'"
        except Exception:
            pass

        return "· ramas evaluadas; sin fusiones requeridas"

    # ---------- INGESTA ASÍNCRONA DE CONVERSACIÓN ----------
    def counts(self) -> dict[str, int]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT (SELECT COUNT(*) FROM cells WHERE archived = 0), "
                "(SELECT COUNT(*) FROM cells WHERE archived = 1), "
                "(SELECT COUNT(*) FROM nodes), (SELECT COUNT(*) FROM nodes WHERE dirty = 1)"
            )
            alive, archived, nodes, dirty = cur.fetchone()
        return {"alive": alive, "archived": archived, "nodes": nodes, "dirty": dirty,
                "ingest_queue": self._ingest_q.qsize()}

    def extract_facts(self, text: str, first: bool = False, origin: str = "") -> tuple[str | None, list[dict[str, Any]] | None]:
        """Extrae hechos atómicos de un texto. Devuelve (fuente_detectada, hechos).
        `hechos` es None si el LLM falló y [] si simplemente no hay nada que memorizar."""
        parts = [
            "Analiza el siguiente texto para incorporarlo al árbol de conocimiento de memoria:",
            "<<<" + str(text)[:6000] + ">>>",
            "",
            "Extrae TODOS los hechos, principios, conceptos clave o datos sustanciales y duraderos.",
            "Reglas fundamentales:",
            "1. Hechos personales: si habla del usuario, redáctalo en tercera persona: 'El usuario...'.",
            "2. Conocimiento técnico/académico: cada concepto como afirmación atómica, autosuficiente y universal.",
            "3. Ignora saludos, preguntas, órdenes, relleno conversacional, botones web, enlaces y números de página. "
            'Si no hay nada que memorizar devuelve "hechos": [].',
            "4. Importancia (1 a 10): 10 = identidad o directivas fundamentales; 7-9 = principios clave; 4-6 = secundarios; 1-3 = menores.",
            "5. Tipo: 'dato' o 'instruccion'.",
        ]
        if "agente" in origin.lower():
            parts.append("6. El texto es una respuesta del agente: registra solo conocimiento verificable, no opiniones ni relleno.")
        if first:
            parts.append("7. Si las primeras líneas son cabecera, título o autores, sintetízalo en 'fuente_detectada'; si no, null.")
            parts.append('Devuelve EXCLUSIVAMENTE este JSON: {"fuente_detectada": "Título y Autores o null", "hechos":[{"texto":"...","importancia":8,"tipo":"dato"}]}')
        else:
            parts.append('Devuelve EXCLUSIVAMENTE este JSON: {"hechos":[{"texto":"...","importancia":8,"tipo":"dato"}]}')
        try:
            obj = self.llm.ask_json(SYSTEM, "\n".join(parts))
        except Exception:
            return None, None
        if not isinstance(obj, dict):
            return None, None
        facts: list[dict[str, Any]] = []
        raw = obj.get("hechos")
        if isinstance(raw, list):
            for f in raw:
                if isinstance(f, str):
                    f = {"texto": f}
                if not isinstance(f, dict):
                    continue
                t = str(f.get("texto", "")).strip()
                if len(t) < 3:
                    continue
                facts.append({
                    "texto": t[:600],
                    "importancia": clamp(f.get("importancia", 6), 1, 10, 6),
                    "tipo": "instruccion" if f.get("tipo") == "instruccion" else "dato",
                })
        src = obj.get("fuente_detectada")
        src = str(src).strip() if src else ""
        return (src if src and src.lower() not in ("null", "none") else None), facts

    def ingest_async(self, text: str, origin: str = "") -> bool:
        """Encola un mensaje de la conversación para extraer hechos y memorizarlos en segundo plano."""
        text = (text or "").strip()
        if len(text) < LTM_MIN_CHARS:
            return False
        is_agent = "agente" in origin.lower()
        if (is_agent and not LTM_INGEST_ASSISTANT) or (not is_agent and not LTM_INGEST_USER):
            return False
        item = (text[:6000], origin or "conversación", time.strftime("%d/%m/%Y a las %H:%M"))
        try:
            self._ingest_q.put_nowait(item)
        except queue.Full:
            self.stats["ingest_dropped"] = self.stats.get("ingest_dropped", 0) + 1
            return False
        self._ensure_ingest_worker()
        return True

    def _ensure_ingest_worker(self) -> None:
        with self._ingest_guard:
            if self._ingest_thread and self._ingest_thread.is_alive():
                return
            self._ingest_stop.clear()
            self._ingest_thread = threading.Thread(target=self._ingest_loop, name="ltm-ingest", daemon=True)
            self._ingest_thread.start()

    def _ingest_loop(self) -> None:
        while not self._ingest_stop.is_set():
            try:
                text, origin, when = self._ingest_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                source = f"Conversación con el usuario ({origin}) el {when}"
                _, facts = self.extract_facts(text, first=False, origin=origin)
                for f in facts or []:
                    imp = max(9.0, f["importancia"]) if f["tipo"] == "instruccion" else f["importancia"]
                    self.store(f["texto"], imp, f["tipo"], meta={"fuente": source})
                    self.stats["ingest_facts"] = self.stats.get("ingest_facts", 0) + 1
            except Exception:
                pass
            finally:
                self._ingest_q.task_done()

    def shutdown(self) -> None:
        self._ingest_stop.set()
        t = self._ingest_thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        try:
            with self.lock:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass

    def close(self) -> None:
        self.shutdown()
        if hasattr(self, "conn"):
            try:
                with self.lock:
                    self.conn.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    # ---------- VISTAS Y EXPLORADOR ----------
    def render(self) -> str:
        lines: list[str] = []
        with self.lock:
            cur = self.conn.cursor()

            def rec(parent_id: int | None, depth: int) -> None:
                cur.execute(
                    "SELECT * FROM nodes WHERE (parent_id IS ? OR parent_id = ?) ORDER BY importance DESC, name ASC",
                    (parent_id, parent_id),
                )
                nodes = cur.fetchall()
                for node in nodes:
                    label = LEVELS[depth] if depth < N else "nodo"
                    lines.append("  " * depth + f"▸ {label}: {node['name']} [imp {node['importance']:.1f}] — {node['description']}")
                    cur.execute("SELECT * FROM cells WHERE node_id = ? AND archived = 0", (node["id"],))
                    for c in cur.fetchall():
                        lines.append("  " * (depth + 1) + f"• ({self.eff(c):.1f}) {c['text']}")
                    if depth < N - 1:
                        rec(node["id"], depth + 1)

            rec(None, 0)
        return "\n".join(lines) if lines else "(memoria vacía)"

    def summary(self) -> str:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cells WHERE archived = 0")
            alive = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM cells WHERE archived = 1")
            archived = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nodes")
            nodes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nodes WHERE dirty = 1")
            dirty = cur.fetchone()[0]
        return (
            f"tick={self.tick} rev={self.rev} | nodos={nodes} (sin describir: {dirty}) | células={alive} | "
            f"archivadas={archived} | motor=SQLite-FTS5 (BM25 nativo) | "
            f"autotests OK={self.stats.get('selftest_ok', 0)} | recall_híbrido={self.stats.get('recall_hibrido', 0)}"
        )

    def get_explorer_tree(self) -> list[dict[str, Any]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, parent_id, level, slug, name, description, importance, celulas, dirty
                FROM nodes
                ORDER BY level ASC, importance DESC, name ASC
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_node_cells(self, node_id: int) -> list[dict[str, Any]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, node_id, text, importance, kind, fuente, ingest_timestamp,
                       created_tick, updated_tick, access_count, archived
                FROM cells
                WHERE node_id = ?
                ORDER BY archived ASC, importance DESC, updated_tick DESC
            """, (node_id,))
            return [dict(r) for r in cur.fetchall()]

    def save_cell(self, cell_id: str | None, node_id: int, text: str, importance: float, kind: str, fuente: str) -> dict[str, Any]:
        with self.lock:
            cur = self.conn.cursor()
            if cell_id:
                cur.execute("""
                    UPDATE cells 
                    SET text = ?, importance = ?, kind = ?, fuente = ?, updated_tick = ?
                    WHERE id = ?
                """, (text.strip(), importance, kind, fuente, self.tick, cell_id))
            else:
                cell_id = f"{self.tick:07d}-{secrets.token_hex(6)}"
                cur.execute("""
                    INSERT INTO cells (id, node_id, text, importance, kind, fuente, ingest_timestamp, created_tick, updated_tick)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (cell_id, node_id, text.strip(), importance, kind, fuente, time.strftime("%Y-%m-%d %H:%M:%S"), self.tick, self.tick))
                cur.execute("UPDATE nodes SET celulas = celulas + 1 WHERE id = ?", (node_id,))
            self.conn.commit()
            self._mark_dirty(node_id)
            self._bump()
            return {"ok": True, "id": cell_id}

    def delete_cell(self, cell_id: str) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT node_id FROM cells WHERE id = ?", (cell_id,))
            row = cur.fetchone()
            if not row:
                return False
            node_id = row[0]
            cur.execute("DELETE FROM cells WHERE id = ?", (cell_id,))
            cur.execute("UPDATE nodes SET celulas = MAX(0, celulas - 1) WHERE id = ?", (node_id,))
            self.conn.commit()
            self._mark_dirty(node_id)
            self._bump()
            return True

    def save_node(self, node_id: int | None, parent_id: int | None, name: str, description: str) -> dict[str, Any]:
        with self.lock:
            cur = self.conn.cursor()
            slug = slugify(name)
            if node_id:
                cur.execute("""
                    UPDATE nodes 
                    SET name = ?, slug = ?, description = ?, dirty = 1
                    WHERE id = ?
                """, (name.strip(), slug, description.strip(), node_id))
            else:
                level = 0
                if parent_id is not None:
                    cur.execute("SELECT level FROM nodes WHERE id = ?", (parent_id,))
                    r = cur.fetchone()
                    level = min(N - 1, r[0] + 1) if r else 0
                cur.execute("""
                    INSERT INTO nodes (parent_id, level, slug, name, description, dirty, created_tick)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (parent_id, level, slug, name.strip(), description.strip(), self.tick))
                node_id = cur.lastrowid
                cur.execute("UPDATE nodes SET pinned = 1 WHERE id = ?", (node_id,))
            self.conn.commit()
            self._bump()
            return {"ok": True, "id": node_id}

    def delete_node(self, node_id: int) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            self.conn.commit()
            self._bump()
            return True

    def export_to_filesystem(self, target_base_dir: Path) -> int:
        target_base_dir = Path(target_base_dir)
        target_base_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM nodes")
            nodes = {r["id"]: dict(r) for r in cur.fetchall()}
            
            def get_full_folder(nid):
                parts = []
                curr = nid
                while curr is not None and curr in nodes:
                    parts.append(nodes[curr]["slug"])
                    curr = nodes[curr]["parent_id"]
                return target_base_dir.joinpath(*reversed(parts))

            for nid, n in nodes.items():
                f_path = get_full_folder(nid)
                f_path.mkdir(parents=True, exist_ok=True)
                n_info = {
                    "name": n["name"], "description": n["description"],
                    "dirty": bool(n["dirty"]), "importance": n["importance"],
                    "created_tick": n["created_tick"], "celulas": n["celulas"]
                }
                (f_path / "_nodo.json").write_text(json.dumps(n_info, ensure_ascii=False, indent=2), encoding="utf-8")

            cur.execute("SELECT * FROM cells")
            for c in cur.fetchall():
                f_path = get_full_folder(c["node_id"]) if not c["archived"] else target_base_dir / "_archivo"
                f_path.mkdir(parents=True, exist_ok=True)
                c_info = dict(c)
                (f_path / f"c-{c['id']}.json").write_text(json.dumps(c_info, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1
        return count
