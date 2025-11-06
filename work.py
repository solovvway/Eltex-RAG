import os
import requests as r
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType
from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json

load_dotenv()  # <-- важно: загружает .env в os.environ

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")

# Выбор провайдера (можно изменить на "yandex" для использования Yandex Cloud)
PROVIDER = "openrouter"  # или "yandex"

if PROVIDER == "openrouter":
    if not OPENROUTER_API_KEY:
        raise RuntimeError("❌ Укажи OPENROUTER_API_KEY в .env")
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
elif PROVIDER == "yandex":
    if not YANDEX_CLOUD_API_KEY:
        raise RuntimeError("❌ Укажи YANDEX_CLOUD_API_KEY в .env")
    if not YANDEX_CLOUD_FOLDER:
        raise RuntimeError("❌ Укажи YANDEX_CLOUD_FOLDER в .env для работы с Yandex Cloud")
    llm_client = OpenAI(
        api_key=YANDEX_CLOUD_API_KEY,
        base_url="https://llm.api.cloud.yandex.net/v1",
        project=YANDEX_CLOUD_FOLDER
    )
else:
    raise RuntimeError("❌ Неизвестный провайдер. Используй 'openrouter' или 'yandex'")
# URLs
doccli = 'https://docs.eltex-co.ru/ede/esr-series-user-manual-firmware-version-1-18-1-380863447.html'
docguide = 'https://docs.eltex-co.ru/ede/esr-series-cli-command-reference-guide-firmware-version-1-13-0-177668705.html'
baseurl = 'https://docs.eltex-co.ru'

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"
# Модели в зависимости от провайдера
if PROVIDER == "openrouter":
    models = [
        "google/gemma-3n-e2b-it:free",
        "deepseek/deepseek-chat-v3.1:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-coder:free",
        "deepseek/deepseek-r1-distill-llama-70b:free"
    ]
else:  # yandex
    models = [
        f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt/latest",
        f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-lite/latest"
    ]

input_dir = "eltex_docs"
supertel_dir = "supertel_docs"

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
    
    # Обработка документов Eltex
    print("📚 Обработка документов Eltex...")
    for filename in os.listdir(input_dir):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        chunks = extract_chunks_from_html(html, f"eltex/{filename}")
        all_chunks.extend(chunks)
        print(f"📄 Обработано: {filename} → {len(chunks)} чанков")
    
    # Обработка документов Supertel
    if os.path.exists(supertel_dir):
        print("📚 Обработка документов Supertel...")
        for filename in os.listdir(supertel_dir):
            if not filename.endswith(".html"):
                continue
            filepath = os.path.join(supertel_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    html = f.read()
                chunks = extract_chunks_from_html(html, f"supertel/{filename}")
                all_chunks.extend(chunks)
                print(f"📄 Обработано: {filename} → {len(chunks)} чанков")
            except Exception as e:
                print(f"⚠️ Ошибка при обработке {filename}: {e}")
    
    return all_chunks  # возвращаем полные чанки с метаданными

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
    # Извлекаем только текст для эмбеддингов
    texts = [ch["text"] for ch in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Создаем точки с полными метаданными
    points = [
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "title": chunk["title"]
            }
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Загружено {len(chunks)} чанков в Qdrant.")
    # --- Гибридный поиск (BM25 + Vector) ---
def query_bm25(question, top_k=5):
    """BM25 текстовый поиск через Qdrant Python client"""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Простой векторный поиск как fallback, если BM25 не работает
        # Это не идеальное решение, но позволит системе работать
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = model.encode([question])[0].tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=top_k,
            with_payload=True
        ).points
        
        return [hit.payload["text"] for hit in hits]
    except Exception as e:
        print(f"⚠️ BM25 error: {e}")
        return []

def query_vector(question, top_k=5):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    q_emb = model.encode([question])[0].tolist()
    hits = client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=top_k).points
    return [hit.payload["text"] for hit in hits]

def chat_with_model(model_name: str, prompt: str, site_url: str = "", site_name: str = ""):
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    completion = llm_client.chat.completions.create(
        extra_headers=headers,
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    response = completion.choices[0].message.content
    print(response)
    return response
def get_key_info():
    """Получает информацию о ключе: остаток, лимит, и др. (только для OpenRouter)"""
    if PROVIDER != "openrouter" or not OPENROUTER_API_KEY:
        return {}
    try:
        resp = r.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
        resp.raise_for_status()
        return resp.json()["data"]
    except Exception:
        return {}

def can_use_free_model():
    """Проверяет возможность использования бесплатных моделей (только для OpenRouter)"""
    if PROVIDER != "openrouter":
        return True  # Для Yandex Cloud проверка не требуется
    info = get_key_info()
    # info содержит: limit, limit_remaining, usage_daily и др.
    # is_free_tier указывает, был ли ранее куплен минимум кредитов
    return info.get("is_free_tier", False)

def chat_with_model_auto(model_name: str, prompt: str, site_url: str = "", site_name: str = ""):
    # проверяем ключ
    key_info = get_key_info()
    rem = key_info.get("limit_remaining")
    # Если нет лимита или он нулевой — нельзя
    if rem is not None and rem <= 0:
        raise RuntimeError("Квота на ключ исчерпана")

    # Отправляем запрос
    try:
        resp = llm_client.chat.completions.create(
            extra_headers={**({"HTTP-Referer": site_url} if site_url else {}),
                           **({"X-Title": site_name} if site_name else {})},
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            # Можно запросить usage вместе с ответом, если подписано в API
            usage={"include": True}
        )
    except HTTPError as e:
        # Если лимит превышен — ошибка 429, можно переключаться
        if e.response.status_code == 429:
            # Можно посмотреть метаданные лимита в заголовках или теле
            # Попробовать использовать другую модель
            print("⚠️ Rate limit reached for model", model_name)
            return None, "rate_limit"
        else:
            raise

    result = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    return result, usage

# Пример логики выбора модели
def ask(prompt):
    # модель по приоритету
    for m in models:
        ans, usage_or_flag = chat_with_model_auto(m, prompt, site_url="", site_name="")
        if ans is not None:
            return ans
        # если rate_limit, пробуем дальше
    # если ни одна модель не вернула — ошибка
    raise RuntimeError("Не удалось найти модель с доступным лимитом")


def pdf_to_html(source_dir="."):
    """Convert PDF files to HTML using unstructured library with improved structure for command documentation"""
    docs = os.listdir(source_dir)
    for doc in docs:
        if not doc.endswith('.pdf'):
            continue
        
        html_filename = doc.replace('.pdf', '.html')
        if html_filename in docs:
            print(f"⏭️  Skipping {doc} - HTML already exists")
            continue
        
        try:
            doc_path = os.path.join(source_dir, doc)
            print(f"📄 Processing {doc}...")
            
            # Partition the PDF file
            elements = partition(filename=doc_path)
            
            # Convert elements to HTML with h1-only structure
            html_parts = []
            current_section = None
            section_content = []
            
            # Add CSS for better structure
            html_parts.append("""
<style>
    .section { margin: 20px 0; }
    .section-content { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
</style>
""")
            
            for element in elements:
                element_type = type(element).__name__
                text = str(element).strip()
                
                if not text:
                    continue
                
                # Check if this is a section heading (numbered, like "6. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ")
                if element_type == "Title" and text[0].isdigit() and '. ' in text[:10]:
                    # Save previous section if exists
                    if current_section and section_content:
                        content_html = "".join(section_content)
                        html_parts.append(f'<div class="section"><h1>{current_section}</h1><div class="section-content">{content_html}</div></div>')
                        section_content = []
                    
                    current_section = text
                else:
                    # Add all other content to the current section
                    section_content.append(f"<p>{text}</p>")
            
            # Save last section if exists
            if current_section and section_content:
                content_html = "".join(section_content)
                html_parts.append(f'<div class="section"><h1>{current_section}</h1><div class="section-content">{content_html}</div></div>')
            
            html_str = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{doc}</title>
</head>
<body>
{''.join(html_parts)}
</body>
</html>"""
            
            output_path = os.path.join(source_dir, html_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            
            print(f"✅ Converted {doc} to {html_filename}")
        except Exception as e:
            print(f"⚠️  Error processing {doc}: {e}")



if __name__ == "__main__":
    # 1. Скачиваем, если нужно
    download_docs_if_needed()

    # 2. Конвертируем PDF в HTML (если есть PDF в supertel_docs)
    if os.path.exists(supertel_dir):
        pdf_to_html(supertel_dir)
    
    # 3. Загружаем и чанкуем из обеих директорий
    chunks = load_all_chunks()
    if not chunks:
        raise RuntimeError("❌ Не удалось извлечь ни одного чанка.")
    
    print(f"📊 Всего извлечено чанков: {len(chunks)}")
    
    # 4. Инициализация модели и Qdrant
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 5. Создаём коллекцию, если её нет
    ensure_collection_exists(client, model, chunks)

    # 6. Пример запроса
    # question = 'create l3vpn with on vrfs TEST1 with rd 1234:123 on router 1.1.1.1 and router 2.2.2.2. Give me 2 configs to this routers'
    question = 'конфигурация supertel nms для создания мониторинга snmp'

    bm25_res = query_bm25(question, 3)
    # print("bm25_res:",bm25_res)
    with open("bm25_res.txt", "w", encoding="utf-8") as f:
        f.write("".join(bm25_res))
    vector_res = query_vector(question, 3)
    with open("vector_res.txt", "w", encoding="utf-8") as f:
        f.write("".join(vector_res))
    # print("vector_res:",vector_res)

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
You are a helpful assistant for network engineer that work with communication equipment. 
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
    output = chat_with_model(models[2],prompt)
    print(output)