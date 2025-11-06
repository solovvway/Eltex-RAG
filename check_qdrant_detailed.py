import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests as r
from collections import Counter

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
        print("=" * 80)
        print("📊 ИНФОРМАЦИЯ О КОЛЛЕКЦИИ")
        print("=" * 80)
        print(f"Название коллекции: {COLLECTION_NAME}")
        print(f"Количество точек (чанков): {collection_info.points_count}")
        print(f"Размерность векторов: {collection_info.config.params.vectors.size}")
        print(f"Метрика расстояния: {collection_info.config.params.vectors.distance}")
        print(f"Статус: {collection_info.status}")
        print()
        
        return collection_info.points_count
    except Exception as e:
        print(f"❌ Ошибка при получении информации о коллекции: {e}")
        print("💡 Возможно, коллекция еще не создана. Запустите work.py сначала.")
        return 0

def check_sources_detailed():
    """Детальная проверка источников данных"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print("=" * 80)
    print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ИСТОЧНИКОВ ДАННЫХ")
    print("=" * 80)
    
    try:
        # Получаем все точки
        all_points = []
        offset = None
        
        print("📥 Загрузка данных из Qdrant...")
        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points, offset = result
            all_points.extend(points)
            
            if offset is None:
                break
        
        print(f"✅ Загружено {len(all_points)} точек\n")
        
        # Анализ источников
        sources = Counter()
        titles_by_source = {"eltex": set(), "supertel": set()}
        
        for point in all_points:
            source_file = point.payload.get("source_file", "unknown")
            title = point.payload.get("title", "No title")
            
            if "eltex" in source_file.lower():
                sources["eltex"] += 1
                titles_by_source["eltex"].add(title)
            elif "supertel" in source_file.lower():
                sources["supertel"] += 1
                titles_by_source["supertel"].add(title)
            else:
                sources["other"] += 1
        
        # Вывод статистики
        print("📊 СТАТИСТИКА ПО ИСТОЧНИКАМ:")
        print("-" * 80)
        for source, count in sources.most_common():
            percentage = (count / len(all_points)) * 100
            print(f"  {source.upper():15} {count:5} чанков ({percentage:5.1f}%)")
        
        print("\n📚 УНИКАЛЬНЫЕ РАЗДЕЛЫ:")
        print("-" * 80)
        
        if titles_by_source["eltex"]:
            print(f"\n  ELTEX ({len(titles_by_source['eltex'])} уникальных разделов):")
            for i, title in enumerate(sorted(list(titles_by_source["eltex"]))[:10], 1):
                print(f"    {i}. {title}")
            if len(titles_by_source["eltex"]) > 10:
                print(f"    ... и еще {len(titles_by_source['eltex']) - 10} разделов")
        
        if titles_by_source["supertel"]:
            print(f"\n  SUPERTEL ({len(titles_by_source['supertel'])} уникальных разделов):")
            for i, title in enumerate(sorted(list(titles_by_source["supertel"]))[:10], 1):
                print(f"    {i}. {title}")
            if len(titles_by_source["supertel"]) > 10:
                print(f"    ... и еще {len(titles_by_source['supertel']) - 10} разделов")
        
        # Примеры чанков
        print("\n📄 ПРИМЕРЫ ЧАНКОВ:")
        print("-" * 80)
        
        eltex_examples = [p for p in all_points if "eltex" in p.payload.get("source_file", "").lower()][:2]
        supertel_examples = [p for p in all_points if "supertel" in p.payload.get("source_file", "").lower()][:2]
        
        if eltex_examples:
            print("\n  ELTEX:")
            for i, point in enumerate(eltex_examples, 1):
                print(f"\n    Пример {i}:")
                print(f"    Файл: {point.payload.get('source_file', 'N/A')}")
                print(f"    Раздел: {point.payload.get('title', 'N/A')}")
                print(f"    Текст: {point.payload.get('text', '')[:150]}...")
        
        if supertel_examples:
            print("\n  SUPERTEL:")
            for i, point in enumerate(supertel_examples, 1):
                print(f"\n    Пример {i}:")
                print(f"    Файл: {point.payload.get('source_file', 'N/A')}")
                print(f"    Раздел: {point.payload.get('title', 'N/A')}")
                print(f"    Текст: {point.payload.get('text', '')[:150]}...")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Ошибка при проверке источников: {e}")

def simulate_query(question, top_k=3):
    """Симулирует запрос к Qdrant с детальным выводом"""
    print("=" * 80)
    print("🔎 СИМУЛЯЦИЯ ЗАПРОСА")
    print("=" * 80)
    print(f"❓ Вопрос: {question}")
    print(f"📊 Количество результатов: {top_k}\n")
    
    # BM25 поиск
    print("─" * 80)
    print("📝 BM25 ПОИСК (лексический - по ключевым словам)")
    print("─" * 80)
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
            bm25_results = result["result"]["points"]
            
            if bm25_results:
                for i, point in enumerate(bm25_results, 1):
                    payload = point["payload"]
                    print(f"\n  Результат {i}:")
                    print(f"  Источник: {payload.get('source_file', 'N/A')}")
                    print(f"  Раздел: {payload.get('title', 'N/A')}")
                    print(f"  Текст: {payload.get('text', '')[:200]}...")
            else:
                print("  ⚠️ Результатов не найдено")
            
            bm25_texts = [p["payload"]["text"] for p in bm25_results]
        else:
            print(f"  ⚠️ Неожиданная структура ответа: {result}")
            bm25_texts = []
    except r.exceptions.HTTPError as e:
        print(f"  ❌ BM25 HTTP error: {e}")
        print(f"     Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        bm25_texts = []
    except Exception as e:
        print(f"  ❌ Ошибка BM25: {e}")
        bm25_texts = []
    
    # Векторный поиск
    print("\n" + "─" * 80)
    print("🧠 ВЕКТОРНЫЙ ПОИСК (семантический - по смыслу)")
    print("─" * 80)
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        q_emb = model.encode([question])[0].tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=top_k,
            with_payload=True
        ).points
        
        if hits:
            for i, hit in enumerate(hits, 1):
                print(f"\n  Результат {i}:")
                print(f"  Релевантность: {hit.score:.4f}")
                print(f"  Источник: {hit.payload.get('source_file', 'N/A')}")
                print(f"  Раздел: {hit.payload.get('title', 'N/A')}")
                print(f"  Текст: {hit.payload.get('text', '')[:200]}...")
        else:
            print("  ⚠️ Результатов не найдено")
        
        vector_texts = [hit.payload["text"] for hit in hits]
    except Exception as e:
        print(f"  ❌ Ошибка векторного поиска: {e}")
        vector_texts = []
    
    # Гибридный результат
    print("\n" + "─" * 80)
    print("🔀 ГИБРИДНЫЙ РЕЗУЛЬТАТ (BM25 + Vector)")
    print("─" * 80)
    seen = set()
    hybrid = []
    
    for t in bm25_texts:
        if t not in seen:
            hybrid.append(("BM25", t))
            seen.add(t)
    
    for t in vector_texts:
        if t not in seen and len(hybrid) < top_k * 2:
            hybrid.append(("Vector", t))
            seen.add(t)
    
    print(f"\n  Найдено уникальных результатов: {len(hybrid)}")
    print(f"  Из них BM25: {sum(1 for source, _ in hybrid if source == 'BM25')}")
    print(f"  Из них Vector: {sum(1 for source, _ in hybrid if source == 'Vector')}")
    
    for i, (source, text) in enumerate(hybrid, 1):
        print(f"\n  Результат {i} [{source}]:")
        print(f"  {text[:200]}...")
    
    print("\n" + "=" * 80)

def main():
    print("\n" + "=" * 80)
    print("🚀 ДЕТАЛЬНАЯ ПРОВЕРКА ДАННЫХ В QDRANT")
    print("=" * 80 + "\n")
    
    # 1. Проверяем информацию о коллекции
    points_count = check_collection_info()
    
    if points_count == 0:
        print("⚠️ Коллекция пуста или не существует!")
        print("💡 Запустите: python work.py")
        return
    
    # 2. Детальная проверка источников
    check_sources_detailed()
    
    # 3. Симулируем запросы
    print("\n" + "=" * 80)
    print("🧪 ТЕСТОВЫЕ ЗАПРОСЫ")
    print("=" * 80 + "\n")
    
    # Запрос 1: Общий запрос про конфигурацию
    simulate_query("configure router interface", top_k=3)
    
    print("\n\n")
    
    # Запрос 2: Специфичный запрос про VPN
    simulate_query("create l3vpn vrf configuration", top_k=3)
    
    print("\n" + "=" * 80)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()