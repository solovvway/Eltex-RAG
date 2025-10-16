import requests as r
from bs4 import BeautifulSoup
import os
import trafilatura
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

docurl = 'https://docs.eltex-co.ru/ede/esr-series-user-manual-firmware-version-1-18-1-380863447.html'
baseurl = 'https://docs.eltex-co.ru'

# Настройки Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"

def getlinks(url):
    res = r.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    links = soup.find_all('a', class_=['accordion-title', 'toggle'])
    res = []
    for link in links:
        if 'href' in link.attrs:
            res.append(link['href'])
    return res

def getdocs(urls):
    for path in urls:
        full_url = baseurl + path
        try:
            res = r.get(full_url)
            filename = path[5:].split('-')[0] + ".html"
            with open(os.path.join('eltex_docs', filename), 'w', encoding='utf-8') as f:
                f.write(res.text)
        except Exception as e:
            print(f"⚠️ Ошибка при загрузке {full_url}: {e}")

def getfilenames():
    files = os.listdir("eltex_docs")
    return [f for f in files if f.endswith(".html")]

def normalize(files):
    for filename in files:
        filepath = os.path.join("eltex_docs", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()

        text = trafilatura.extract(html, include_comments=False, include_tables=False)

        if text:
            outpath = os.path.join("norm_docs", filename.replace(".html", ".txt"))
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ {outpath}")
        else:
            print(f"⚠️ Не удалось извлечь: {filename}")

def create_chunks(files):
    all_chunks = []
    for file in files:
        chunks = []
        current_chunk = []
        filepath = os.path.join("norm_docs", file.replace(".html", ".txt"))
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding="utf-8") as text:
            lines = text.readlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            current_chunk.append(stripped)

            if not stripped.startswith("esr"):
                if current_chunk and any(l.startswith("esr") for l in current_chunk[:-1]):
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current_chunk = []

        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        all_chunks.extend(chunks)
    return all_chunks

def build_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        dim = model.get_sentence_embedding_dimension()
        from qdrant_client.models import TextIndexParams, TokenizerType

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            # Включаем full-text индекс для поля "text"
            hnsw_config=None,  # можно оставить по умолчанию
        )

        # Добавляем текстовый индекс для BM25
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,  # или "whitespace", "multilingual"
                min_token_len=2,
                max_token_len=15,
                lowercase=True
            )
        )

        # Проверим, что индекс действительно создан
        indexes = client.get_collection(COLLECTION_NAME).payload_schema
        print("📑 Индексы в коллекции:")
        for field_name, schema in indexes.items():
            print(f" - {field_name}: {schema}")


        print(f"🆕 Коллекция {COLLECTION_NAME} создана с BM25-индексом")

        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        points = [
            PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Загружено {len(chunks)} чанков")
    else:
        print(f"📂 Коллекция уже существует")

    return model, client

import requests

def query_bm25(question, client=None, top_k=5):
    try:
        resp = requests.post(
            f"http://localhost:6333/collections/{COLLECTION_NAME}/points/query",
            json={
                "limit": top_k,
                "query": {"text": question},
                "using": "text"
            }
        )

        resp.raise_for_status()
        points = resp.json()["result"]["points"]
        return [(p["payload"]["text"], "BM25") for p in points]
    except Exception as e:
        print(f"⚠️ BM25 search error: {e}")
        return []

def query_vector(question, model, client, top_k=5):
    """Возвращает список кортежей: (text, 'Vector')"""
    try:
        q_emb = model.encode([question], convert_to_numpy=True)[0].tolist()
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,  # вектор передаём как query
            using=None,   # None = используем основной вектор
            limit=top_k
        )
        return [(hit.payload["text"], "Vector") for hit in search_result.points]
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return []


def hybrid_query(question, model, client, bm25_k=5, vector_k=5, total_k=10):
    """
    Гибридный поиск: BM25 + Vector.
    Убирает дубликаты, сохраняя приоритет BM25.
    Возвращает список (text, source)
    """
    bm25_results = query_bm25(question, client, bm25_k)
    vector_results = query_vector(question, model, client, vector_k)

    seen = set()
    hybrid = []

    # Сначала добавляем BM25 (приоритет!)
    for text, src in bm25_results:
        if text not in seen:
            hybrid.append((text, src))
            seen.add(text)

    # Потом — векторные, которых ещё нет
    for text, src in vector_results:
        if text not in seen and len(hybrid) < total_k:
            hybrid.append((text, src))
            seen.add(text)

    # Обрезаем до total_k
    return hybrid[:total_k]


def query_rag(question, model, client, top_k=10):
    q_emb = model.encode([question], convert_to_numpy=True)[0].tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb,
        limit=top_k
    )
    results = [hit.payload["text"] for hit in search_result]
    return results

def rag_answer(question, model, client, top_k=6, apiurl="http://localhost:8080/v1/chat/completions"):
    hybrid_results = hybrid_query(
        question=question,
        model=model,
        client=client,
        bm25_k=3,
        vector_k=3,
        total_k=top_k
    )

    context_lines = []
    for i, (text, source) in enumerate(hybrid_results, 1):
        context_lines.append(f"[{source}] Chunk #{i}:\n{text}\n{'-'*50}")

    context = "\n\n".join(context_lines)
    with open("prompt.txt", "w") as file:
        file.write(context)

    prompt = f"""
You are a helpful assistant for network engineer that work with Eltex routers. 
Answer the question using the context below.
If answer is not found, say 'not enough data'.
Answer briefly and to the point. Answer questions primarily using configuration commands, writing them down in ```.

For example, for the question ‘How do I configure the access port on the switch?’, give the answer

Context:
{context}

Question: {question}
"""

    payload = {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant working with vendor documentation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    # resp = r.post(apiurl, json=payload)
    # resp.raise_for_status()
    # data = resp.json()
    return prompt


# from qdrant_client import QdrantClient
import csv

client = QdrantClient(host="localhost", port=6333)
collection_name = "eltex_docs"

# Получаем все точки (по частям, если их много)
all_points = []
offset = None
while True:
    records, offset = client.scroll(
        collection_name=collection_name,
        limit=1000,  # можно увеличить
        offset=offset,
        with_payload=True,
        with_vectors=False  # не выгружаем векторы, только текст
    )
    all_points.extend(records)
    if offset is None:
        break

# Сохраняем в CSV
with open("qdrant_export.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text"])
    for point in all_points:
        writer.writerow([point.id, point.payload.get("text", "")])

print(f"✅ Экспортировано {len(all_points)} записей в qdrant_export.csv")


# === Основной поток ===
# input_dir = "eltex_docs"
# output_dir = "norm_docs"
# os.makedirs(input_dir, exist_ok=True)
# os.makedirs(output_dir, exist_ok=True)

# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# client.delete_collection("eltex_docs")

# # Раскомментируй при первом запуске:
# links = getlinks(docurl)
# getdocs(links)
# files = getfilenames()
# normalize(files)

# files = getfilenames()
# chunks = create_chunks(files)
# print("Пример чанка:", chunks[1] if chunks else "Нет чанков")

# model, client = build_vectorstore(chunks)


# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# # question = input(">")
# question = 'configure mpls l2vpn in compella mode'
# results = rag_answer(question, model, client)
# print(results)