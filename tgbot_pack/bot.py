import os
import re
import logging
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, TokenizerType
from openai import OpenAI
from requests.exceptions import HTTPError

# === ENV ===
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not API_KEY or not BOT_TOKEN:
    raise RuntimeError("❌ Укажи OPENROUTER_API_KEY и TELEGRAM_BOT_TOKEN в .env")

# === OpenRouter client ===
llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# === Qdrant ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"

# === Глобальные переменные ===
user_model_choice = {}  # {user_id: model_name}
cached_free_models = []


# === Получение и сортировка бесплатных моделей ===
def get_free_models():
    """Получить и отсортировать бесплатные модели по размеру."""
    global cached_free_models
    if cached_free_models:
        return cached_free_models

    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])

    free_models = [m["id"] for m in data if m.get("id", "").endswith(":free")]
    # сортировка по размеру модели (70b > 33b > 20b > 7b > 3b > 1b)
    def extract_size(model_id):
        match = re.search(r"(\d+)(b|B)", model_id)
        return int(match.group(1)) if match else 0

    free_models.sort(key=extract_size, reverse=True)
    cached_free_models = free_models
    return free_models


# === Проверка квоты / ключа ===
def get_key_info():
    resp = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {API_KEY}"})
    resp.raise_for_status()
    return resp.json().get("data", {})


# === Работа с Qdrant ===
def query_bm25(question, top_k=5):
    try:
        resp = requests.post(
            f"http://localhost:6333/collections/{COLLECTION_NAME}/points/query",
            json={"limit": top_k, "query": {"text": question}, "using": "text"}
        )
        resp.raise_for_status()
        return [p["payload"]["text"] for p in resp.json()["result"]["points"]]
    except Exception:
        return []


def query_vector(question, top_k=5):
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    q_emb = emb_model.encode([question])[0].tolist()
    hits = client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=top_k).points
    return [h.payload["text"] for h in hits]


# === Работа с моделями ===
def chat_with_model_safe(model_name: str, prompt: str):
    """Отправляет запрос, возвращает ответ или None при rate_limit."""
    try:
        resp = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except HTTPError as e:
        if e.response.status_code == 429:
            return None  # квота исчерпана
        raise


async def ask_with_fallback(update: Update, model_name: str, prompt: str):
    """Пробует выбранную модель, при лимите — переключается на следующую."""
    free_models = get_free_models()

    if model_name not in free_models:
        await update.message.reply_text(f"⚠️ Модель {model_name} недоступна. Переключаюсь автоматически.")
        model_name = free_models[0]
        user_model_choice[update.message.from_user.id] = model_name

    idx = free_models.index(model_name)
    for i in range(idx, len(free_models)):
        current_model = free_models[i]
        response = chat_with_model_safe(current_model, prompt)
        print(prompt,'\n',response,'\n')
        if response:
            if i != idx:
                await update.message.reply_text(
                    f"⚠️ Модель {model_name} недоступна (лимит), переключаюсь на {current_model}"
                )
                user_model_choice[update.message.from_user.id] = current_model
            return response
    return "❌ Все бесплатные модели исчерпали лимит. Попробуй позже."


# === Telegram Bot ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    free_models = get_free_models()
    keyboard = [[InlineKeyboardButton(m, callback_data=m)] for m in free_models]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 Привет! Выбери модель для работы:", reply_markup=reply_markup)


async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model = query.data
    user_model_choice[query.from_user.id] = model
    await query.edit_message_text(f"✅ Выбрана модель: {model}\nТеперь отправь вопрос.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    model = user_model_choice.get(user_id)

    if not model:
        await update.message.reply_text("⚠️ Сначала выбери модель с помощью /start")
        return

    question = update.message.text.strip()
    await update.message.reply_text("🔎 Ищу в документации...")

    bm25_res = query_bm25(question, 3)
    vector_res = query_vector(question, 3)

    hybrid = []
    seen = set()
    for t in bm25_res + vector_res:
        if t not in seen:
            hybrid.append(t)
            seen.add(t)
    context_text = "\n\n---\n\n".join(hybrid)

    prompt = f"""
You are a helpful assistant for network engineer that work with Eltex routers. 
Answer the question using the context below.
If answer is not found, say 'not enough data'.
Answer briefly and to the point. Use configuration commands.
Question: {question}

Context:
{context_text}
"""

    await update.message.reply_text("🧠 Отправляю запрос модели...")
    response = await ask_with_fallback(update, model, prompt)
    await update.message.reply_text(response)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(select_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
