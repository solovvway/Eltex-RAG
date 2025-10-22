from flask import Flask, request, Response
from flask_cors import CORS
import os
import requests as r
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType
from openai import OpenAI
import json

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"
input_dir = "eltex_docs"
os.makedirs(input_dir, exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
doccli = 'https://docs.eltex-co.ru/ede/esr-series-user-manual-firmware-version-1-18-1-380863447.html'
docguide = 'https://docs.eltex-co.ru/ede/esr-series-cli-command-reference-guide-firmware-version-1-13-0-177668705.html'
baseurl = 'https://docs.eltex-co.ru'

# Functions from provided code
def getlinks(url):
    res = r.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    links = soup.find_all('a', class_=['accordion-title', 'toggle'])
    return [link['href'] for link in links if 'href' in link.attrs]

def download_docs_if_needed():
    if os.listdir(input_dir):
        return
    for doc_url in [doccli, docguide]:
        links = getlinks(doc_url)
        for path in links:
            full_url = baseurl + path
            try:
                res = r.get(full_url)
                filename = path.lstrip('/').replace('/', '_').replace('.html', '') + ".html"
                with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(res.text)
            except:
                pass

def extract_chunks_from_html(html_content, source_file):
    soup = BeautifulSoup(html_content, "html.parser")
    main = soup.find("div", {"id": "main-content"}) or soup
    headings = main.find_all(["h2", "h1"])
    if not headings:
        text = trafilatura.extract(html_content, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 20:
            return [{"source_file": source_file, "title": "Document", "text": text.strip()}]
        return []
    chunks = []
    for i, h in enumerate(headings):
        title = h.get_text(strip=True)
        if not title:
            continue
        content_parts = []
        sibling = h.next_sibling
        while sibling:
            if sibling.name in ["h1", "h2"]:
                break
            if hasattr(sibling, 'get_text'):
                txt = sibling.get_text(strip=True)
                if txt:
                    content_parts.append(txt)
            elif isinstance(sibling, str):
                txt = sibling.strip()
                if txt:
                    content_parts.append(txt)
            sibling = sibling.next_sibling
        chunk_text = "\n".join(content_parts).strip()
        full_text = f"{title}\n\n{chunk_text}".strip()
        if len(full_text) < 20:
            continue
        chunks.append({
            "source_file": source_file,
            "title": title,
            "text": full_text
        })
    return chunks

def load_all_chunks():
    all_chunks = []
    for filename in os.listdir(input_dir):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        chunks = extract_chunks_from_html(html, filename)
        all_chunks.extend(chunks)
    return [ch["text"] for ch in all_chunks]

def ensure_collection_exists():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        return
    dim = model.get_sentence_embedding_dimension()
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=15,
            lowercase=True
        )
    )
    chunks = load_all_chunks()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

def query_bm25(question, top_k=3):
    try:
        resp = r.post(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/points/query",
            json={"limit": top_k, "query": {"text": question}, "using": "text"}
        )
        resp.raise_for_status()
        return [p["payload"]["text"] for p in resp.json()["result"]["points"]]
    except:
        return []

def query_vector(question, top_k=3):
    q_emb = model.encode([question])[0].tolist()
    hits = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=top_k).points
    return [hit.payload["text"] for hit in hits]

def make_prompt(question):
    bm25_res = query_bm25(question, 3)
    vector_res = query_vector(question, 3)
    seen = set()
    hybrid = []
    for t in bm25_res:
        if t not in seen:
            hybrid.append(t)
            seen.add(t)
    for t in vector_res:
        if t not in seen and len(hybrid) < 6:
            hybrid.append(t)
            seen.add(t)
    context = "\n\n---\n\n".join(hybrid)
    return f"""
You are a helpful assistant for network engineers working with Eltex routers.
Answer using the context below. If no answer is found, say 'not enough data'.
Answer briefly with configuration commands in ```.
Question: {question}

Context:
{context}
"""

# Initialize RAG
download_docs_if_needed()
ensure_collection_exists()

# API Route
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    if not messages:
        return Response(json.dumps({"error": "No messages provided"}), status=400, mimetype='application/json')
    
    question = messages[-1]['content']
    prompt = make_prompt(question)
    
    completion = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    def generate():
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content.encode('utf-8')
    
    return Response(generate(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)