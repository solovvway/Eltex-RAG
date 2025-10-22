import os
import requests as r
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)
# URLs
doccli = 'https://docs.eltex-co.ru/ede/esr-series-user-manual-firmware-version-1-18-1-380863447.html'
docguide = 'https://docs.eltex-co.ru/ede/esr-series-cli-command-reference-guide-firmware-version-1-13-0-177668705.html'
baseurl = 'https://docs.eltex-co.ru'

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"
models = [
    "google/gemma-3n-e2b-it:free",
    "deepseek/deepseek-chat-v3.1:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-r1-distill-llama-70b:free"
]

input_dir = "eltex_docs"
os.makedirs(input_dir, exist_ok=True)

# === Функции ===

def getlinks(url):
    res = r.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    links = soup.find_all('a', class_=['accordion-title', 'toggle'])
    return [link['href'] for link in links if 'href' in link.attrs]

def download_docs_if_needed():
    if os.listdir(input_dir):
        print("📁 Документы уже загружены. Пропускаем скачивание.")
        return

    print("📥 Загрузка документации...")
    for doc_url in [doccli, docguide]:
        links = getlinks(doc_url)
        for path in links:
            full_url = baseurl + path
            try:
                res = r.get(full_url)
                filename = path.lstrip('/').replace('/', '_').replace('.html', '') + ".html"
                with open(os.path.join(input_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(res.text)
                print(f"✅ Загружено: {filename}")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке {full_url}: {e}")

def extract_chunks_from_html(html_content, source_file):
    soup = BeautifulSoup(html_content, "html.parser")
    main = soup.find("div", {"id": "main-content"}) or soup

    # Сначала пробуем h2, если нет — h1
    headings = main.find_all(["h2", "h1"])
    if not headings:
        # fallback: весь текст как один чанк
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
        print(f"📄 Обработано: {filename} → {len(chunks)} чанков")
    return [ch["text"] for ch in all_chunks]  # только текст для эмбеддингов

def ensure_collection_exists(client, model, chunks):
    if client.collection_exists(COLLECTION_NAME):
        print(f"📂 Коллекция {COLLECTION_NAME} уже существует.")
        return
    # client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # client.delete_collection("eltex_docs")
    print(f"🆕 Создаём коллекцию {COLLECTION_NAME}...")
    dim = model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    client.create_payload_index(
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

    print("🧠 Генерация эмбеддингов...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Загружено {len(chunks)} чанков в Qdrant.")
    # --- Гибридный поиск (BM25 + Vector) ---
def query_bm25(question, top_k=5):
    try:
        resp = r.post(
            f"http://localhost:6333/collections/{COLLECTION_NAME}/points/query",
            json={"limit": top_k, "query": {"text": question}, "using": "text"}
        )
        resp.raise_for_status()
        return [p["payload"]["text"] for p in resp.json()["result"]["points"]]
    except Exception as e:
        print(f"⚠️ BM25 error: {e}")
        return []

def query_vector(question, top_k=5):
    q_emb = model.encode([question])[0].tolist()
    hits = client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=top_k).points
    return [hit.payload["text"] for hit in hits]

def chat_with_model(model_name: str, prompt: str, site_url: str = "", site_name: str = ""):
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    completion = client.chat.completions.create(
        extra_headers=headers,
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    response = completion.choices[0].message.content
    print(response)
    return response

def initRAG():
    download_docs_if_needed()
    chunks = load_all_chunks()
    if not chunks:
        raise RuntimeError("❌ Не удалось извлечь ни одного чанка.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection_exists(client, model, chunks)

def make_prompt(question: str) -> str:
    bm25_res = query_bm25(question, 3)
    vector_res = query_vector(question, 3)
    seen = set()
    hybrid = []
    for t in bm25_res:
        if t not in seen:
            hybrid.append(t); seen.add(t)
    for t in vector_res:
        if t not in seen and len(hybrid) < 6:
            hybrid.append(t); seen.add(t)

    context = "\n\n---\n\n".join(hybrid)

    prompt = f"""
You are a helpful assistant for network engineer that work with Eltex routers. 
Answer the question using the context below.
If answer is not found, say 'not enough data'.
Answer briefly and to the point. Answer questions primarily using configuration commands, writing them down in ```.
Question: {question}

Context:
{context}
"""
    return prompt

if __name__ == "__main__":
    # 1. Скачиваем, если нужно
    # download_docs_if_needed()

    # 2. Загружаем и чанкуем
    chunks = load_all_chunks()


    # 3. Инициализация модели и Qdrant
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 4. Создаём коллекцию, если её нет
    ensure_collection_exists(client, model, chunks)

    # 5. Пример запроса
    question = 'create l3vpn with on vrfs TEST1 with rd 1234:123 on router 1.1.1.1 and router 2.2.2.2. Give me 2 configs to this routers'

    bm25_res = query_bm25(question, 3)
    vector_res = query_vector(question, 3)

    seen = set()
    hybrid = []
    for t in bm25_res:
        if t not in seen:
            hybrid.append(t); seen.add(t)
    for t in vector_res:
        if t not in seen and len(hybrid) < 6:
            hybrid.append(t); seen.add(t)

    context = "\n\n---\n\n".join(hybrid)

    prompt = f"""
You are a helpful assistant for network engineer that work with Eltex routers. 
Answer the question using the context below.
If answer is not found, say 'not enough data'.
Answer briefly and to the point. Answer questions primarily using configuration commands, writing them down in ```.
Question: {question}

Context:
{context}
"""

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    print("✅ Промпт сохранён в prompt.txt")
    output = chat_with_model(models[0],prompt)
    print(output)