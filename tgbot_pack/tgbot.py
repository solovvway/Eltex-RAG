# bot.py
import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from dotenv import load_dotenv

from openai import OpenAI
from work import (
    query_bm25,
    query_vector,
    chat_with_model_auto,
    models,  # список моделей
)
load_dotenv()  # <-- важно: загружает .env в os.environ

# === Настройки ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # токен телеграм-бота
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

logging.basicConfig(level=logging.INFO)
user_model_choice = {}  # {user_id: model_name}


# === Команды ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(m, callback_data=m)] for m in models
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет! 👋 Я бот для работы с Eltex RAG.\n"
        "Выбери модель:", reply_markup=reply_markup
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model_name = query.data
    user_model_choice[query.from_user.id] = model_name
    await query.edit_message_text(f"✅ Модель выбрана: {model_name}\nТеперь отправь мне вопрос.")


# === Обработка сообщений ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    question = update.message.text.strip()

    model_name = user_model_choice.get(user_id)
    if not model_name:
        await update.message.reply_text("⚠️ Сначала выбери модель с помощью /start")
        return

    await update.message.reply_text("🔎 Выполняю поиск в документации...")

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

    context_text = "\n\n---\n\n".join(hybrid)
    prompt = f"""
You are a helpful assistant for network engineer that work with Eltex routers. 
Answer the question using the context below.
If answer is not found, say 'not enough data'.
Answer briefly and to the point. Answer questions primarily using configuration commands, writing them down in ```.
Question: {question}

Context:
{context_text}
"""

    await update.message.reply_text("🧠 Отправляю запрос модели...")
    try:
        response = chat_with_model_auto(model_name, prompt)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка при обращении к модели:\n{e}")


# === Запуск ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
