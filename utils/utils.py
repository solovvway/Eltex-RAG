from qdrant_client import QdrantClient
import csv


def dump():
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

def delete():
    client = QdrantClient(host="localhost", port=6333)
    client.delete_collection("eltex_docs")

if __name__ == "__main__":
    # dump()
    # delete()
    pass