"""RAG bits: reading the documents, making vectors, searching them, asking the model."""

import json
import math
import os
import re
import sqlite3
import subprocess
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "rag.db")

TOP_K = 3                 # how many chunks we feed to the model
CHUNK_TARGET_CHARS = 700  # approximate chunk size

NO_ANSWER = "I don't have that information in my documents."

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the context "
    "passages given below. Keep the answer to 2-4 sentences and do not add any file "
    "names or references - the application shows the sources itself.\n"
    "If the passages do not contain the answer, reply with exactly this sentence and "
    "nothing else: " + NO_ANSWER
)

STOPWORDS = set("""a an the and or but if then than that this these those is are was were be been being
of in on at to for from with without by as it its into about over under again further once here there
all any both each few more most other some such no nor not only own same so too very can will just do
does did doing you your yours he she they them his her their we our us i me my what which who whom how
when where why""".split())


def chunk_text(text):
    """Cut a document into chunks, roughly a paragraph or two each."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > CHUNK_TARGET_CHARS:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# Foundry Local has no embedding model in the catalog here, so I make the
# vectors myself with TF-IDF. Only build_vocabulary() and embed() would have
# to change to use a real embedding model.
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 1 and t not in STOPWORDS]


def build_vocabulary(chunks):
    """Count in how many chunks each word shows up, and turn that into IDF weights."""
    df = {}
    for c in chunks:
        for term in set(tokenize(c)):
            df[term] = df.get(term, 0) + 1
    n = max(len(chunks), 1)
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def embed(text, idf):
    """Make a vector out of a piece of text: {word: weight}, length normalised to 1."""
    tf = {}
    for term in tokenize(text):
        if term in idf:
            tf[term] = tf.get(term, 0) + 1
    vec = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def cosine(a, b):
    """Both vectors are already normalised, so the dot product is the cosine."""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def connect():
    return sqlite3.connect(DB_PATH)


def ingest():
    """Read documents/, chunk it all up, vectorise it and write it to the database."""
    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    files = sorted(f for f in os.listdir(DOCS_DIR)
                   if f.lower().endswith((".txt", ".md")) and not f.startswith("."))

    records = []  # (source, chunk_text)
    for name in files:
        with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as fh:
            for chunk in chunk_text(fh.read()):
                records.append((name, chunk))

    idf = build_vocabulary([c for _, c in records])

    db = connect()
    db.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE chunks (
            id        INTEGER PRIMARY KEY,
            source    TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL   -- vector as JSON, SQLite has no vector type
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    db.executemany(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        [(src, txt, json.dumps(embed(txt, idf))) for src, txt in records],
    )
    db.execute("INSERT INTO meta (key, value) VALUES ('idf', ?)", (json.dumps(idf),))
    db.commit()
    db.close()
    return {"documents": len(files), "chunks": len(records), "vocabulary": len(idf)}


def stats():
    if not os.path.exists(DB_PATH):
        return {"documents": 0, "chunks": 0}
    db = connect()
    try:
        row = db.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM chunks").fetchone()
        return {"chunks": row[0], "documents": row[1]}
    except sqlite3.Error:
        return {"documents": 0, "chunks": 0}
    finally:
        db.close()


def get_top_chunks(query, k=TOP_K):
    """Find the k chunks closest to the question. Best one first."""
    if not os.path.exists(DB_PATH):
        return []
    db = connect()
    try:
        idf = json.loads(db.execute("SELECT value FROM meta WHERE key='idf'").fetchone()[0])
        rows = db.execute("SELECT source, content, embedding FROM chunks").fetchall()
    except (sqlite3.Error, TypeError):
        return []
    finally:
        db.close()

    qvec = embed(query, idf)
    if not qvec:
        return []

    scored = [(cosine(qvec, json.loads(emb)), src, content) for src, content, emb in rows]
    scored.sort(reverse=True, key=lambda r: r[0])
    return [{"score": round(s, 4), "source": src, "content": txt}
            for s, src, txt in scored[:k] if s > 0.02]


_cache = {"endpoint": None, "model": None}


def foundry_endpoint():
    """Get the address of the local Foundry service, start it if it is down."""
    if _cache["endpoint"]:
        return _cache["endpoint"]
    if os.environ.get("FOUNDRY_ENDPOINT"):
        _cache["endpoint"] = os.environ["FOUNDRY_ENDPOINT"].rstrip("/")
        return _cache["endpoint"]
    try:
        out = subprocess.run(["foundry", "service", "status"], capture_output=True,
                             text=True, timeout=60).stdout
        if "not running" in out.lower():
            out = subprocess.run(["foundry", "service", "start"], capture_output=True,
                                 text=True, timeout=120).stdout
        match = re.search(r"http://[\d.]+:\d+", out)
        if match:
            _cache["endpoint"] = match.group(0)
    except (OSError, subprocess.SubprocessError):
        pass
    return _cache["endpoint"]


def foundry_model():
    """Pick whichever chat model is available locally. Phi first if there is one."""
    if _cache["model"]:
        return _cache["model"]
    if os.environ.get("FOUNDRY_MODEL"):
        _cache["model"] = os.environ["FOUNDRY_MODEL"]
        return _cache["model"]
    endpoint = foundry_endpoint()
    if not endpoint:
        return None
    try:
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=30) as resp:
            ids = [m["id"] for m in json.loads(resp.read())["data"]]
    except (OSError, ValueError, KeyError):
        return None
    for wanted in ("phi-4-mini-instruct", "phi", "qwen"):
        for mid in ids:
            if wanted in mid.lower():
                _cache["model"] = mid
                return mid
    _cache["model"] = ids[0] if ids else None
    return _cache["model"]


def ask_model(messages, timeout=300):
    """Send the prompt to the model running on this machine."""
    endpoint, model = foundry_endpoint(), foundry_model()
    if not endpoint or not model:
        raise RuntimeError("Foundry Local is not available. Run: foundry service start")

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


# search -> put the chunks in the prompt -> let the model write the answer
def answer_query(question):
    chunks = get_top_chunks(question)
    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "model": foundry_model()}

    context = "\n\n".join(
        "SOURCE: {}\n{}".format(c["source"], c["content"]) for c in chunks
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Context passages:\n\n{}\n\nQuestion: {}".format(context, question)},
    ]
    answer = ask_model(messages)

    # The model kept citing the wrong file (about 1 answer in 3), so I throw away
    # any file name it writes. The sources come from the search step instead.
    answer = re.sub(r"\s*\[[^\]\n]{0,60}\.(txt|md)\]", "", answer).strip()

    # If it said it doesn't know, the chunks were not used, so don't list them.
    if answer.lower().startswith("i don't have that information"):
        return {"answer": NO_ANSWER, "sources": [], "model": foundry_model()}

    return {"answer": answer, "sources": chunks, "model": foundry_model()}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        print("Indexed:", ingest())
    else:  # console chat, handy for testing without the browser
        print("Local RAG assistant - type a question (Ctrl+C to quit)\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            result = answer_query(q)
            print("\nAssistant:", result["answer"])
            if result["sources"]:
                print("Sources:", ", ".join(sorted({s["source"] for s in result["sources"]})))
            print()
