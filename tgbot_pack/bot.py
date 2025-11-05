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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("❌ Укажи TELEGRAM_BOT_TOKEN в .env")

if not OPENROUTER_API_KEY and not YANDEX_CLOUD_API_KEY:
    raise RuntimeError("❌ Укажи хотя бы один из ключей: OPENROUTER_API_KEY или YANDEX_CLOUD_API_KEY в .env")

if YANDEX_CLOUD_API_KEY and not YANDEX_CLOUD_FOLDER:
    raise RuntimeError("❌ Укажи YANDEX_CLOUD_FOLDER в .env для работы с Yandex Cloud")

# === Qdrant ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"

# === Глобальные переменные ===
user_model_choice = {}  # {user_id: model_name}
user_provider_choice = {}  # {user_id: "openrouter" | "yandex"}
user_llm_client = {}  # {user_id: OpenAI client instance}
user_models_list = {}  # {user_id: [list of models]} - для хранения списка моделей пользователя
cached_free_models = []
cached_yandex_models = []


# === Получение моделей OpenRouter ===
def get_free_models():
    """Получить и отсортировать бесплатные модели OpenRouter по размеру."""
    global cached_free_models
    if cached_free_models:
        return cached_free_models

    if not OPENROUTER_API_KEY:
        return []

    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
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
    except Exception as e:
        print(f"⚠️ Ошибка получения моделей OpenRouter: {e}")
        return []


# === Получение моделей Yandex Cloud ===
def get_yandex_models():
    """Получить список доступных моделей Yandex Cloud."""
    global cached_yandex_models
    if cached_yandex_models:
        return cached_yandex_models

    if not YANDEX_CLOUD_API_KEY or not YANDEX_CLOUD_FOLDER:
        return []

    try:
        # Используем OpenAI client для получения списка моделей
        temp_client = OpenAI(
            api_key=YANDEX_CLOUD_API_KEY,
            base_url="https://llm.api.cloud.yandex.net/v1",
            project=YANDEX_CLOUD_FOLDER
        )
        models_list = temp_client.models.list()
        
        # Извлекаем ID моделей и форматируем их
        models = []
        for model in models_list.data:
            model_id = model.id
            # Форматируем в правильный URI для Yandex Cloud
            if not model_id.startswith("gpt://"):
                model_uri = f"gpt://{YANDEX_CLOUD_FOLDER}/{model_id}"
            else:
                model_uri = model_id
            models.append(model_uri)
        
        if not models:
            # Дефолтные модели Yandex Cloud
            models = [
                f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt/latest",
                f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-lite/latest"
            ]
        
        cached_yandex_models = models
        return cached_yandex_models
    except Exception as e:
        print(f"⚠️ Ошибка получения моделей Yandex Cloud: {e}")
        # Возвращаем дефолтные модели
        cached_yandex_models = [
            f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt/latest",
            f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-lite/latest"
        ]
        return cached_yandex_models


# === Проверка квоты / ключа OpenRouter ===
def get_key_info():
    if not OPENROUTER_API_KEY:
        return {}
    try:
        resp = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception:
        return {}


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
def chat_with_model_safe(user_id: int, model_name: str, prompt: str):
    """Отправляет запрос, возвращает ответ или None при rate_limit."""
    llm_client = user_llm_client.get(user_id)
    if not llm_client:
        return None
    
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
    except Exception as e:
        print(f"⚠️ Ошибка при запросе к модели: {e}")
        return None


async def ask_with_fallback(update: Update, model_name: str, prompt: str):
    """Пробует выбранную модель, при лимите — переключается на следующую (только для OpenRouter)."""
    user_id = update.message.from_user.id
    provider = user_provider_choice.get(user_id, "openrouter")
    
    # Для Yandex Cloud не делаем fallback
    if provider == "yandex":
        response = chat_with_model_safe(user_id, model_name, prompt)
        if response:
            return response
        return "❌ Ошибка при обращении к Yandex Cloud. Попробуй позже."
    
    # Для OpenRouter делаем fallback
    free_models = get_free_models()
    
    if model_name not in free_models:
        await update.message.reply_text(f"⚠️ Модель {model_name} недоступна. Переключаюсь автоматически.")
        model_name = free_models[0] if free_models else None
        if not model_name:
            return "❌ Нет доступных моделей."
        user_model_choice[user_id] = model_name

    idx = free_models.index(model_name)
    for i in range(idx, len(free_models)):
        current_model = free_models[i]
        response = chat_with_model_safe(user_id, current_model, prompt)
        print(prompt,'\n',response,'\n')
        if response:
            if i != idx:
                await update.message.reply_text(
                    f"⚠️ Модель {model_name} недоступна (лимит), переключаюсь на {current_model}"
                )
                user_model_choice[user_id] = current_model
            return response
    return "❌ Все бесплатные модели исчерпали лимит. Попробуй позже."


# === Telegram Bot ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выбор провайдера при старте."""
    user_id = update.message.from_user.id
    
    # Определяем доступные провайдеры
    available_providers = []
    if OPENROUTER_API_KEY:
        available_providers.append(("🌐 OpenRouter", "openrouter"))
    if YANDEX_CLOUD_API_KEY:
        available_providers.append(("☁️ Yandex Cloud", "yandex"))
    
    if not available_providers:
        await update.message.reply_text("❌ Нет доступных провайдеров. Проверь настройки API ключей.")
        return
    
    # Если только один провайдер доступен, выбираем его автоматически
    if len(available_providers) == 1:
        provider = available_providers[0][1]
        user_provider_choice[user_id] = provider
        
        # Инициализируем клиент
        if provider == "openrouter":
            user_llm_client[user_id] = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        else:
            user_llm_client[user_id] = OpenAI(
                api_key=YANDEX_CLOUD_API_KEY,
                base_url="https://llm.api.cloud.yandex.net/v1",
                project=YANDEX_CLOUD_FOLDER
            )
        
        await show_model_selection(update, provider)
        return
    
    # Показываем выбор провайдера
    keyboard = [[InlineKeyboardButton(name, callback_data=f"provider:{code}")] for name, code in available_providers]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 Привет! Выбери провайдера для работы:", reply_markup=reply_markup)


async def show_model_selection(update: Update, provider: str):
    """Показывает выбор модели для выбранного провайдера."""
    # Получаем user_id
    if hasattr(update, 'callback_query') and update.callback_query:
        user_id = update.callback_query.from_user.id
    else:
        user_id = update.message.from_user.id
    
    if provider == "openrouter":
        models = get_free_models()
        if not models:
            msg = "❌ Не удалось получить список моделей OpenRouter."
            if hasattr(update, 'callback_query') and update.callback_query:
                await update.callback_query.edit_message_text(msg)
            else:
                await update.message.reply_text(msg)
            return
    else:  # yandex
        models = get_yandex_models()
        if not models:
            msg = "❌ Не удалось получить список моделей Yandex Cloud."
            if hasattr(update, 'callback_query') and update.callback_query:
                await update.callback_query.edit_message_text(msg)
            else:
                await update.message.reply_text(msg)
            return
    
    # Сохраняем список моделей для пользователя
    user_models_list[user_id] = models
    
    # Создаем кнопки с короткими индексами вместо полных URI
    keyboard = []
    for idx, model in enumerate(models):
        # Для Yandex показываем короткое имя, для OpenRouter - полное
        if provider == "yandex":
            # Извлекаем короткое имя из URI (например, yandexgpt/latest)
            display_name = model.split('/')[-2] + '/' + model.split('/')[-1] if '/' in model else model
        else:
            display_name = model
        
        # Используем короткий индекс в callback_data
        keyboard.append([InlineKeyboardButton(display_name, callback_data=f"model:{idx}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.edit_message_text(
            f"✅ Провайдер: {'OpenRouter' if provider == 'openrouter' else 'Yandex Cloud'}\n\nВыбери модель для работы:",
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            f"✅ Провайдер: {'OpenRouter' if provider == 'openrouter' else 'Yandex Cloud'}\n\nВыбери модель для работы:",
            reply_markup=reply_markup
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data
    
    if data.startswith("provider:"):
        # Выбор провайдера
        provider = data.split(":", 1)[1]
        user_provider_choice[user_id] = provider
        
        # Инициализируем клиент
        if provider == "openrouter":
            user_llm_client[user_id] = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        else:
            user_llm_client[user_id] = OpenAI(
                api_key=YANDEX_CLOUD_API_KEY,
                base_url="https://llm.api.cloud.yandex.net/v1",
                project=YANDEX_CLOUD_FOLDER
            )
        
        await show_model_selection(update, provider)
    
    elif data.startswith("model:"):
        # Выбор модели по индексу
        model_idx = int(data.split(":", 1)[1])
        
        # Получаем список моделей пользователя
        models_list = user_models_list.get(user_id, [])
        if not models_list or model_idx >= len(models_list):
            await query.edit_message_text("❌ Ошибка: модель не найдена. Попробуй /start снова.")
            return
        
        model = models_list[model_idx]
        user_model_choice[user_id] = model
        
        provider = user_provider_choice.get(user_id, "openrouter")
        provider_name = "OpenRouter" if provider == "openrouter" else "Yandex Cloud"
        
        # Показываем короткое имя модели
        if provider == "yandex":
            display_name = model.split('/')[-2] + '/' + model.split('/')[-1] if '/' in model else model
        else:
            display_name = model
        
        await query.edit_message_text(
            f"✅ Провайдер: {provider_name}\n✅ Модель: {display_name}\n\nТеперь отправь вопрос."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    model = user_model_choice.get(user_id)
    provider = user_provider_choice.get(user_id)

    if not model or not provider:
        await update.message.reply_text("⚠️ Сначала выбери провайдера и модель с помощью /start")
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
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Бот запущен.")
    app.run_polling()


if __name__ == "__main__":
    main()
