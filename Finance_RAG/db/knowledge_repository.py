from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from Finance_RAG.parser_chunk_search.chunker import KnowledgeFile
from Finance_RAG.settings import Settings


DB_PATH = Path(Settings.basic_settings.DB_ROOT_PATH)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_name TEXT UNIQUE NOT NULL,
            kb_info TEXT,
            file_count INTEGER DEFAULT 0,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_ext TEXT,
            kb_name TEXT NOT NULL,
            is_parsed INTEGER DEFAULT 0,
            file_version INTEGER DEFAULT 1,
            file_mtime REAL DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(kb_name, file_name)
        );

        CREATE TABLE IF NOT EXISTS experiment_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exp_name TEXT UNIQUE NOT NULL,
            kb_name TEXT NOT NULL,
            text_splitter_name TEXT,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            embed_model TEXT,
            vs_type TEXT,
            index_type TEXT,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS file_doc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_name TEXT NOT NULL,
            file_name TEXT NOT NULL,
            exp_name TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            meta_data TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_file_doc_exp_file ON file_doc(exp_name, file_name);
        CREATE INDEX IF NOT EXISTS idx_file_doc_doc_id ON file_doc(doc_id);
        """
    )
    conn.commit()


def kb_exists(kb_name: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM knowledge_base WHERE lower(kb_name)=lower(?)",
            (kb_name,),
        ).fetchone()
        return row is not None


def add_kb_to_db(kb_name: str, kb_info: str):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO knowledge_base(kb_name, kb_info)
            VALUES(?, ?)
            ON CONFLICT(kb_name) DO UPDATE SET kb_info=excluded.kb_info
            """,
            (kb_name, kb_info),
        )
        conn.commit()
    return True


def experiment_exists(exp_name: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM experiment_config WHERE lower(exp_name)=lower(?)",
            (exp_name,),
        ).fetchone()
        return row is not None


def list_experiments_from_db(kb_name: str) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT exp_name FROM experiment_config WHERE lower(kb_name)=lower(?)",
            (kb_name,),
        ).fetchall()
        return [row["exp_name"] for row in rows]


def delete_experiment_from_db(exp_name: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM file_doc WHERE exp_name=?", (exp_name,))
        cursor = conn.execute("DELETE FROM experiment_config WHERE exp_name=?", (exp_name,))
        conn.commit()
        return cursor.rowcount > 0


def list_docs_from_db(exp_name: str, file_name: str = None) -> List[Dict]:
    with _connect() as conn:
        if file_name:
            rows = conn.execute(
                "SELECT * FROM file_doc WHERE exp_name=? AND file_name=?",
                (exp_name, file_name),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM file_doc WHERE exp_name=?", (exp_name,)).fetchall()

        return [
            {
                "kb_name": row["kb_name"],
                "exp_name": row["exp_name"],
                "file_name": row["file_name"],
                "doc_id": row["doc_id"],
                "metadata": json.loads(row["meta_data"] or "{}"),
            }
            for row in rows
        ]


def add_experiment_to_db(
    exp_name: str,
    kb_name: str,
    vs_type: str,
    embed_model: str,
    chunk_size: int,
    index_type: str,
    text_splitter_name: str = "RecursiveChineseBlockSplitter",
    chunk_overlap: int = 0,
):
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO experiment_config(
                exp_name, kb_name, text_splitter_name, chunk_size, chunk_overlap,
                embed_model, vs_type, index_type
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_name,
                kb_name,
                text_splitter_name,
                chunk_size,
                chunk_overlap,
                embed_model,
                vs_type,
                index_type,
            ),
        )
        conn.commit()
    return True


def file_exists_in_kb(kb_name: str, file_name: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM knowledge_file WHERE lower(kb_name)=lower(?) AND lower(file_name)=lower(?)",
            (kb_name, file_name),
        ).fetchone()
        return row is not None


def add_file_to_db(kb_file: KnowledgeFile, is_parsed: bool = False):
    with _connect() as conn:
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()
        existed = file_exists_in_kb(kb_file.kb_name, kb_file.filename)
        conn.execute(
            """
            INSERT INTO knowledge_file(
                file_name, file_ext, kb_name, is_parsed, file_mtime, file_size
            )
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(kb_name, file_name) DO UPDATE SET
                is_parsed=excluded.is_parsed,
                file_mtime=excluded.file_mtime,
                file_size=excluded.file_size,
                file_version=file_version + 1
            """,
            (kb_file.filename, kb_file.ext, kb_file.kb_name, int(is_parsed), mtime, size),
        )
        if not existed:
            conn.execute(
                "UPDATE knowledge_base SET file_count=file_count + 1 WHERE kb_name=?",
                (kb_file.kb_name,),
            )
        conn.commit()
    return True


def delete_file_from_db(kb_file: KnowledgeFile):
    with _connect() as conn:
        existed = file_exists_in_kb(kb_file.kb_name, kb_file.filename)
        conn.execute(
            "DELETE FROM knowledge_file WHERE kb_name=? AND file_name=?",
            (kb_file.kb_name, kb_file.filename),
        )
        conn.execute(
            "DELETE FROM file_doc WHERE kb_name=? AND file_name=?",
            (kb_file.kb_name, kb_file.filename),
        )
        if existed:
            conn.execute(
                "UPDATE knowledge_base SET file_count=max(file_count - 1, 0) WHERE kb_name=?",
                (kb_file.kb_name,),
            )
        conn.commit()
    return True


def add_chunk_to_db_by_expName(
    kb_name: str,
    exp_name: str,
    file_name: str,
    doc_infos: List[Dict],
):
    if not doc_infos:
        return False
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO file_doc(kb_name, exp_name, file_name, doc_id, meta_data)
            VALUES(?, ?, ?, ?, ?)
            """,
            [
                (
                    kb_name,
                    exp_name,
                    file_name,
                    doc["id"],
                    json.dumps(doc.get("metadata", {}), ensure_ascii=False),
                )
                for doc in doc_infos
            ],
        )
        conn.commit()
    return True


def delete_chunk_from_db_by_expName_fileName(
    exp_name: str,
    file_name: str = None,
) -> List[Dict]:
    docs = list_docs_from_db(exp_name=exp_name, file_name=file_name)
    with _connect() as conn:
        if file_name:
            conn.execute("DELETE FROM file_doc WHERE exp_name=? AND file_name=?", (exp_name, file_name))
        else:
            conn.execute("DELETE FROM file_doc WHERE exp_name=?", (exp_name,))
        conn.commit()
    return docs


def list_chunkId_from_db_by_expName_fileName(
    exp_name: str,
    file_name: str,
) -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM file_doc WHERE exp_name=? AND file_name=?",
            (exp_name, file_name),
        ).fetchall()
        return [row["doc_id"] for row in rows]
