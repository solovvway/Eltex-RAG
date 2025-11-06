#!/usr/bin/env python3
"""
Скрипт для пересоздания коллекции Qdrant с новыми метаданными
"""
import os
from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"

def rebuild_collection():
    """Удаляет и пересоздает коллекцию"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print("=" * 80)
    print("🔄 ПЕРЕСОЗДАНИЕ КОЛЛЕКЦИИ QDRANT")
    print("=" * 80)
    
    # Проверяем существование коллекции
    if client.collection_exists(COLLECTION_NAME):
        print(f"\n📂 Коллекция '{COLLECTION_NAME}' существует")
        
        # Получаем информацию
        info = client.get_collection(COLLECTION_NAME)
        print(f"   Количество точек: {info.points_count}")
        
        # Удаляем
        print(f"\n🗑️  Удаление коллекции '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        print("   ✅ Коллекция удалена")
    else:
        print(f"\n📂 Коллекция '{COLLECTION_NAME}' не существует")
    
    print("\n" + "=" * 80)
    print("📝 СЛЕДУЮЩИЕ ШАГИ:")
    print("=" * 80)
    print("\n1. Запустите основной скрипт для загрузки данных:")
    print("   python work.py")
    print("\n2. После загрузки проверьте данные:")
    print("   python check_qdrant_detailed.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        rebuild_collection()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Убедитесь, что Qdrant запущен:")
        print("   docker-compose up -d")