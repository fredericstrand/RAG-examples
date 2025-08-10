import os
import json
import uuid
from pathlib import Path
import fitz
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv(find_dotenv())

INPUT_DIR = os.environ.get("INPUT_DIR", "data")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))
LLAMA_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "dbname": os.environ.get("POSTGRES_DB", "graph_rag"),
    "user": os.environ.get("POSTGRES_USER", "fredericstrand"),
    "password": os.environ.get("POSTGRES_PASSWORD", "supersecret"),
}

def make_pool(pg: dict) -> SimpleConnectionPool:
    pool = SimpleConnectionPool(
        1, 10,
        host=pg["host"],
        port=pg["port"],
        user=pg["user"],
        password=pg["password"],
        dbname=pg["dbname"],
        connect_timeout=5,
        application_name="graph_rag_loader",
    )
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_database(), current_user, inet_server_addr(), inet_server_port();")
            cur.fetchone()
    finally:
        pool.putconn(conn)
    return pool

pool = make_pool(PG)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "neo4j")

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv"}

def mk_uuid() -> uuid.UUID:
    return uuid.uuid4()


def embed_texts(client: OpenAI, texts: List[str], batch_size: int = 128) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out

def parse_with_llamaparse(input_dir: str, max_chars: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.pdf"))):
        doc = fitz.open(path)
        title = (doc.metadata or {}).get("title") or os.path.basename(path)
        file_name = os.path.basename(path)
        page_count = len(doc)
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue
            if len(text) <= max_chars:
                chunks.append({
                    "file_name": file_name,
                    "title": title,
                    "page_count": page_count,
                    "page_number": i,
                    "chunk_index": 0,
                    "text": text,
                })
            else:
                start = 0
                idx = 0
                while start < len(text):
                    end = start + max_chars
                    segment = text[start:end]
                    if not segment.strip():
                        break
                    chunks.append({
                        "file_name": file_name,
                        "title": title,
                        "page_count": page_count,
                        "page_number": i,
                        "chunk_index": idx,
                        "text": segment.strip(),
                    })
                    idx += 1
                    if end >= len(text):
                        break
                    start = max(0, end - overlap)
        doc.close()
    return chunks

def ensure_document(cur, doc_row):
    cur.execute(
        """
        INSERT INTO documents (id, file_name, title, page_count)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
          file_name = EXCLUDED.file_name,
          title = EXCLUDED.title,
          page_count = EXCLUDED.page_count
        """,
        (str(doc_row["id"]), doc_row["file_name"], doc_row["title"], doc_row["page_count"]),
    )

def insert_chunk(cur, chunk_row):
    vec = "[" + ",".join(f"{x:.8f}" for x in chunk_row["embedding"]) + "]"
    cur.execute(
        """
        INSERT INTO chunks (id, doc_id, page_number, chunk_index, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            str(chunk_row["id"]),
            str(chunk_row["doc_id"]),
            chunk_row["page_number"],
            chunk_row["chunk_index"],
            chunk_row["text"],
            vec,
        ),
    )

def neo4j_upsert_document(session, doc_row):
    session.run(
        """
        MERGE (d:Document {id: $id})
        SET d.file_name = $file_name,
            d.title = $title,
            d.page_count = $page_count
        """,
        id=str(doc_row["id"]),
        file_name=doc_row["file_name"],
        title=doc_row["title"],
        page_count=doc_row["page_count"],
    )

def neo4j_upsert_page_and_chunk(session, doc_row, chunk_row):
    session.run(
        """
        MATCH (d:Document {id: $doc_id})
        MERGE (p:Page {doc_id: $doc_id, number: $page_number})
          ON CREATE SET p.created_at = timestamp()
        MERGE (d)-[:HAS_PAGE]->(p)
        MERGE (c:Chunk {doc_id: $doc_id, page_number: $page_number, chunk_index: $chunk_index})
          ON CREATE SET c.created_at = timestamp()
        SET c.text = $text,
            c.length = $length
        MERGE (p)-[:HAS_CHUNK {index: $chunk_index}]->(c)
        """,
        doc_id=str(doc_row["id"]),
        page_number=chunk_row["page_number"],
        chunk_index=chunk_row["chunk_index"],
        text=chunk_row["text"],
        length=len(chunk_row["text"]),
    )
