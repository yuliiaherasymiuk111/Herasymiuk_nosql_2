
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
EMBEDDINGS_FILE = BASE_DIR / "embeddings" / "embeddings.npy"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "arxiv-semantic-search")

DIMENSION = 768
BATCH_SIZE = 50


def load_to_pinecone():
    if not API_KEY:
        print("Помилка: PINECONE_API_KEY не знайдено у .env")
        return

    if not DATA_FILE.exists() or not EMBEDDINGS_FILE.exists():
        print("Помилка: відсутній датасет або файл ембеддингів.")
        return

    df = pd.read_parquet(DATA_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(df) != len(embeddings):
        print("Помилка: кількість статей і векторів не збігається.")
        return

    if embeddings.ndim != 2 or embeddings.shape[1] != DIMENSION:
        print("Помилка: неправильна розмірність ембеддингів.")
        return

    print(f"Завантажено статей: {len(df)}")
    print(f"Форма ембеддингів: {embeddings.shape}")

    pc = Pinecone(api_key=API_KEY)

    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        print(f"Створюю індекс {INDEX_NAME}...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    print("Очікую готовності індексу...")

    while True:
        description = pc.describe_index(INDEX_NAME)

        if description.status["ready"]:
            break

        time.sleep(5)

    index = pc.Index(INDEX_NAME)

    print("Завантажую вектори в Pinecone...")

    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = []

        for i in range(start, min(start + BATCH_SIZE, len(df))):
            row = df.iloc[i]

            arxiv_id = str(row["id"])

            metadata = {
                "arxiv_id": arxiv_id,
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            }

            batch.append(
                {
                    "id": arxiv_id,
                    "values": embeddings[i].tolist(),
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=batch)

    print("\nЗавантаження завершено!")

    stats = index.describe_index_stats()

    print(
        f"Кількість векторів у Pinecone: "
        f"{stats.total_vector_count}"
    )


if __name__ == "__main__":
    load_to_pinecone()
