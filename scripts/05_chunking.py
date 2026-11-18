
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "chunking_results.csv"

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"

INDEX_NAME = "arxiv-semantic-search"

SAMPLE_SIZE = 30
CHUNK_SIZE = 100
OVERLAP = 20
BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 50
TOP_K = 5

QUERIES = [
    "machine learning methods for image classification",
    "black holes and gravitational waves",
    "quantum computing and quantum information",
]


def fixed_size_chunks(text):
    """Розбиваю текст на чанки по 100 слів із перекриттям."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_SIZE])

        if chunk:
            chunks.append(chunk)

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def semantic_chunks(text):
    """Групую речення, щоб не розривати їх між чанками."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


def create_embeddings(texts, tokenizer, model, device):
    """Створюю нормалізовані ембеддинги."""
    vectors = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            embeddings = mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )

        vectors.append(embeddings.cpu().numpy())

    embeddings = np.vstack(vectors).astype(np.float32)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / np.maximum(norms, 1e-12)


def prepare_chunks(df, chunk_function, strategy):
    """Готую чанки та їхні метадані."""
    records = []

    for _, row in df.iterrows():
        # Для chunking використовую саме анотацію
        text = str(row["abstract"]).strip()
        chunks = chunk_function(text)

        for chunk_number, chunk_text in enumerate(chunks):
            records.append(
                {
                    "id": (
                        f"{strategy}_"
                        f"{row['id']}_"
                        f"{chunk_number}"
                    ),
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"]),
                    "text": chunk_text,
                    "chunk_number": chunk_number,
                    "year": int(row["year"]),
                    "category": str(row["category"]),
                }
            )

    return records


def upload_chunks(index, records, embeddings, namespace):
    """Завантажую чанки в окремий namespace."""
    vectors = []

    for record, embedding in zip(records, embeddings):
        vectors.append(
            {
                "id": record["id"],
                "values": embedding.tolist(),
                "metadata": {
                    "arxiv_id": record["arxiv_id"],
                    "title": record["title"],
                    "chunk_text": record["text"][:2000],
                    "chunk_number": record["chunk_number"],
                    "year": record["year"],
                    "category": record["category"],
                },
            }
        )

    for start in tqdm(
        range(0, len(vectors), UPSERT_BATCH_SIZE),
        desc=f"Завантаження {namespace}",
    ):
        batch = vectors[start:start + UPSERT_BATCH_SIZE]

        index.upsert(
            vectors=batch,
            namespace=namespace,
        )


def search_chunks(index, query, query_vector, namespace):
    """Шукаю найбільш схожі чанки в Pinecone."""
    response = index.query(
        vector=query_vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
    )

    results = []

    print(f"\nЗапит: {query}")
    print(f"Namespace: {namespace}")

    for rank, match in enumerate(response.matches, start=1):
        metadata = match.metadata or {}

        title = metadata.get("title", "")
        chunk_text = metadata.get("chunk_text", "")
        score = float(match.score)

        print(f"\n{rank}. {title}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Текст: {chunk_text[:250]}...")

        results.append(
            {
                "strategy": namespace,
                "query": query,
                "rank": rank,
                "article_id": metadata.get("arxiv_id", ""),
                "title": title,
                "chunk_number": metadata.get("chunk_number", 0),
                "similarity": round(score, 4),
                "chunk_text": chunk_text,
            }
        )

    return results


def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "Не знайдено PINECONE_API_KEY у файлі .env"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Не знайдено датасет: {DATA_FILE}"
        )

    df = pd.read_parquet(DATA_FILE)

    # Обираю 30 статей із найдовшими анотаціями
    df = (
        df.assign(
            abstract_length=df["abstract"].fillna("").str.len()
        )
        .sort_values("abstract_length", ascending=False)
        .head(SAMPLE_SIZE)
        .reset_index(drop=True)
    )

    print(f"Статей для експерименту: {len(df)}")
    print(
        "Середня довжина анотації:",
        round(df["abstract_length"].mean()),
        "символів",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Пристрій: {device}")

    print("\nЗавантажую SPECTER2...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter_name = model.load_adapter(
        ADAPTER_NAME,
        source="hf",
        set_active=True,
    )

    model.set_active_adapters(adapter_name)
    model.to(device)
    model.eval()

    pc = Pinecone(api_key=api_key)

    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(
            f"Індекс {INDEX_NAME} не існує!"
        )

    index = pc.Index(INDEX_NAME)

    strategies = {
        "chunks-fixed": fixed_size_chunks,
        "chunks-semantic": semantic_chunks,
    }

    all_results = []

    # Створюю ембеддинги для тестових запитів один раз
    query_vectors = create_embeddings(
        QUERIES,
        tokenizer,
        model,
        device,
    )

    for namespace, chunk_function in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Стратегія: {namespace}")

        records = prepare_chunks(
            df,
            chunk_function,
            namespace,
        )

        print(f"Кількість чанків: {len(records)}")

        texts = [record["text"] for record in records]

        chunk_embeddings = create_embeddings(
            texts,
            tokenizer,
            model,
            device,
        )

        upload_chunks(
            index,
            records,
            chunk_embeddings,
            namespace,
        )

        for query, query_vector in zip(QUERIES, query_vectors):
            results = search_chunks(
                index,
                query,
                query_vector,
                namespace,
            )

            all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nЕксперимент завершено!")
    print(f"Результатів пошуку: {len(results_df)}")
    print(f"CSV збережено: {OUTPUT_FILE}")

    print("\nПеревірка Pinecone:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
