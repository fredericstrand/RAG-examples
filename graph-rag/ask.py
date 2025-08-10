import os
import time
from typing import List
from dotenv import load_dotenv, find_dotenv
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from openai import OpenAI

load_dotenv(find_dotenv())

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKFILL = os.environ.get("BACKFILL", "0") in ("1", "true", "True", "yes", "YES")
TABLE = os.environ.get("CHUNKS_TABLE", "public.chunks")

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
    return pool

pool = make_pool(PG)
client = OpenAI(api_key=OPENAI_API_KEY)
chat_history: List[str] = []

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def ensure_index():
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON {TABLE} USING hnsw (embedding vector_cosine_ops);")
            conn.commit()
    finally:
        pool.putconn(conn)

def backfill_chunks(batch_size: int = 128, max_batches: int | None = None):
    while True:
        if max_batches is not None and max_batches <= 0:
            break
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT id, text FROM {TABLE} WHERE embedding IS NULL LIMIT %s;", (batch_size,))
            rows = cur.fetchall()
            if not rows:
                break
            ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            vecs = [d.embedding for d in resp.data]
            payload = [("[" + ",".join(f"{x:.6f}" for x in v) + "]", i) for v, i in zip(vecs, ids)]
            psycopg2.extras.execute_batch(
                cur,
                f"UPDATE {TABLE} SET embedding = %s::vector WHERE id = %s",
                payload,
                page_size=100,
            )
            conn.commit()
            if max_batches is not None:
                max_batches -= 1
        finally:
            pool.putconn(conn)

def retrieve_context(question: str, k: int = 5) -> str:
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        emb = get_embedding(question)
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        sql = f"""
            SELECT
                text AS content,
                doc_id,
                page_number,
                chunk_index
            FROM {TABLE}
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (vec_literal, k))
        rows = cur.fetchall()
        if not rows:
            return ""
        parts = []
        for content, doc_id, page, idx in rows:
            parts.append(f"[doc:{doc_id} p.{page} chunk:{idx}]\n{content}")
        return "\n\n---\n\n".join(parts)
    finally:
        pool.putconn(conn)

def build_prompt(context, chat_history, question):
    history_formatted = "\n".join(chat_history) if chat_history else ""
    return (
        "You are 'LawAI', a professional AI assistant specializing in Norwegian labour law (arbeidsrett). "
        "Your purpose is to assist with clear, legally accurate, and practically useful information strictly based on the provided context. "
        "You must always respond in Norwegian unless the user explicitly requests another language.\n\n"
        "## ROLE AND USAGE\n"
        "You act as an information assistant for HR professionals, managers, and other users who are not legal experts. "
        "You provide structured, accessible legal information, not legal advice. "
        "You do not write or generate any legally binding documents, nor do you speculate beyond what the provided context contains.\n\n"
        "## INSTRUCTIONS\n"
        "1. Begin with a short and neutral introduction that frames the topic.\n"
        "2. Use clear bullet points when helpful, but do not use formatting like bold, italic, or headings.\n"
        "3. All explanations must be grounded entirely in the 'Context' section below. You may NOT reference legal content that is not explicitly included in the Context.\n"
        "4. You may explain a paragraph (§) only if it is explicitly present or quoted in the Context. Do not infer or assume its content.\n"
        "5. If a paragraph or legal concept is mentioned in the question but not in the context, clearly state that you cannot explain it due to missing information.\n"
        "6. Use only the context as your legal source – not your own internal legal knowledge, prior responses, or common knowledge.\n"
        "7. You may use the conversation history to maintain tone and style, but not as a source of legal information.\n"
        "8. Always include a final section titled 'Kilder' (Sources), where you list only the specific paragraphs and laws actually present in the Context.\n"
        "## LANGUAGE AND STYLE\n"
        "* Write in clear, plain Norwegian suitable for a general audience.\n"
        "* Maintain a professional, informative, and neutral tone.\n"
        "* Avoid legal jargon unless it is explained simply and clearly.\n"
        "* Use natural language and coherent reasoning. If you cannot answer, state this clearly and briefly.\n\n"
        f"# Context:\n{context}\n\n"
        f"# Previous conversation:\n{history_formatted}\n\n"
        f"# User question:\n{question}\n\n"
        "Svar:"
    )

def ask(question: str, k: int = 5) -> str:
    context = retrieve_context(question, k)
    prompt = build_prompt(context, chat_history, question)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Du er en hjelpsom assistent med ekspertise på norsk jus."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5000,
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    chat_history.append(f"Bruker: {question}")
    chat_history.append(f"Assistent: {answer}")
    return answer

def main():
    if BACKFILL:
        ensure_index()
        backfill_chunks(batch_size=128)
    while True:
        try:
            question = input("Spørsmål (skriv 'exit' for å avslutte): ")
        except (EOFError, KeyboardInterrupt):
            print("\nAvslutter.")
            break
        if question.lower() == "exit":
            break
        answer = ask(question)
        print("\nSvar:", answer, "\n")

if __name__ == "__main__":
    main()
