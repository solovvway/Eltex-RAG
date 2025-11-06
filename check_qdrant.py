import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests as r

# Настройки
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "eltex_docs"

def check_collection_info():
    """Проверяет информацию о коллекции"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        # Получаем информацию о коллекции
        collection_info = client.get_collection(COLLECTION_NAME)
        print("=" * 60)
        print("📊 ИНФОРМАЦИЯ О КОЛЛЕКЦИИ")
        print("=" * 60)
        print(f"Название: {COLLECTION_NAME}")
        print(f"Количество точек: {collection_info.points_count}")
        print(f"Размерность векторов: {collection_info.config.params.vectors.size}")
        print(f"Метрика расстояния: {collection_info.config.params.vectors.distance}")
        print()
        
        return collection_info.points_count
    except Exception as e:
        print(f"❌ Ошибка при получении информации о коллекции: {e}")
        return 0

def check_sources():
    """Проверяет наличие данных из разных источников"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print("=" * 60)
    print("🔍 ПРОВЕРКА ИСТОЧНИКОВ ДАННЫХ")
    print("=" * 60)
    
    # Получаем все точки (ограничим до 1000 для примера)
    try:
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        eltex_count = 0
        supertel_count = 0
        other_count = 0
        
        eltex_samples = []
        supertel_samples = []
        
        for point in points:
            text = point.payload.get("text", "")
            
            # Проверяем по содержимому (так как source_file не сохраняется в payload)
            # Можно улучшить, добавив source_file в payload при загрузке
            if "eltex" in text.lower() or "esr" in text.lower():
                eltex_count += 1
                if len(eltex_samples) < 2:
                    eltex_samples.append(text[:100] + "...")
            elif "supertel" in text.lower() or "таиц" in text.lower():
                supertel_count += 1
                if len(supertel_samples) < 2:
                    supertel_samples.append(text[:100] + "...")
            else:
                other_count += 1
        
        print(f"📄 Документы Eltex: ~{eltex_count} чанков")
        if eltex_samples:
            print("   Примеры:")
            for i, sample in enumerate(eltex_samples, 1):
                print(f"   {i}. {sample}")
        
        print(f"\n📄 Документы Supertel: ~{supertel_count} чанков")
        if supertel_samples:
            print("   Примеры:")
            for i, sample in enumerate(supertel_samples, 1):
                print(f"   {i}. {sample}")
        
        print(f"\n📄 Другие документы: ~{other_count} чанков")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка при проверке источников: {e}")

def simulate_query(question, top_k=3):
    """Симулирует запрос к Qdrant"""
    print("=" * 60)
    print("🔎 СИМУЛЯЦИЯ ЗАПРОСА")
    print("=" * 60)
    print(f"Вопрос: {question}\n")
    
    # BM25 поиск
    print("📝 BM25 поиск (лексический):")
    try:
        resp = r.post(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/points/query",
            json={
                "query": {"text": question},
                "using": "text",
                "limit": top_k,
                "with_payload": True
            }
        )
        resp.raise_for_status()
        result = resp.json()
        
        if "result" in result and "points" in result["result"]:
            bm25_results = [p["payload"]["text"] for p in result["result"]["points"]]
            for i, text in enumerate(bm25_results, 1):
                print(f"\n{i}. {text[:200]}...")
        else:
            print(f"⚠️ Неожиданная структура ответа: {result}")
            bm25_results = []
    except Exception as e:
        print(f"❌ Ошибка BM25: {e}")
        bm25_results = []
    
    # Векторный поиск
    print("\n" + "=" * 60)
    print("🧠 Векторный поиск (семантический):")
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        q_emb = model.encode([question])[0].tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=top_k
        ).points
        
        vector_results = [hit.payload["text"] for hit in hits]
        
        for i, result in enumerate(vector_results, 1):
            score = hits[i-1].score if i <= len(hits) else 0
            print(f"\n{i}. [Score: {score:.4f}] {result[:200]}...")
    except Exception as e:
        print(f"❌ Ошибка векторного поиска: {e}")
        vector_results = []
    
    # Гибридный результат
    print("\n" + "=" * 60)
    print("🔀 ГИБРИДНЫЙ РЕЗУЛЬТАТ (BM25 + Vector):")
    seen = set()
    hybrid = []
    for t in bm25_results:
        if t not in seen:
            hybrid.append(t)
            seen.add(t)
    for t in vector_results:
        if t not in seen and len(hybrid) < top_k * 2:
            hybrid.append(t)
            seen.add(t)
    
    print(f"\nНайдено уникальных результатов: {len(hybrid)}")
    for i, result in enumerate(hybrid, 1):
        print(f"\n{i}. {result[:200]}...")
    
    print("\n" + "=" * 60)

def main():
    print("\n🚀 ПРОВЕРКА ДАННЫХ В QDRANT\n")
    
    # 1. Проверяем информацию о коллекции
    points_count = check_collection_info()
    
    if points_count == 0:
        print("⚠️ Коллекция пуста или не существует!")
        return
    
    # 2. Проверяем источники данных
    check_sources()
    
    # 3. Симулируем запросы
    print("\n" + "=" * 60)
    print("ТЕСТОВЫЕ ЗАПРОСЫ")
    print("=" * 60 + "\n")
    
    # Запрос 1: Общий запрос про конфигурацию
    simulate_query("configure router interface", top_k=3)
    
    print("\n\n")
    
    # Запрос 2: Специфичный запрос про VPN
    simulate_query("create l3vpn vrf configuration", top_k=3)

if __name__ == "__main__":
    main()