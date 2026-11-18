
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
EMBEDDINGS_FILE = BASE_DIR / "embeddings" / "embeddings.npy"
OUTPUT_FILE = BASE_DIR / "data" / "metrics_comparison.csv"

TOP_K = 5


def print_results(metric_name, indices, scores, df):
    print(f"\n--- {metric_name} ---")

    for rank, idx in enumerate(indices, start=1):
        article = df.iloc[idx]

        print(f"\n{rank}. {article['title']}")
        print(f"   ID: {article['id']}")
        print(f"   Категорія: {article['category']}")
        print(f"   Рік: {article['year']}")
        print(f"   Значення метрики: {scores[idx]:.6f}")


def main():
    if not DATA_FILE.exists():
        print(f"Не знайдено датасет: {DATA_FILE}")
        return

    if not EMBEDDINGS_FILE.exists():
        print(f"Не знайдено ембеддинги: {EMBEDDINGS_FILE}")
        return

    df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(df) != len(embeddings):
        raise ValueError(
            "Кількість статей і ембеддингів не збігається!"
        )

    print(f"Завантажено статей: {len(df)}")
    print(f"Форма ембеддингів: {embeddings.shape}")

    # Перевіряю нормалізацію
    norms = np.linalg.norm(embeddings, axis=1)

    print(f"Середня норма векторів: {norms.mean():.6f}")

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            "Ембеддинги не нормалізовані!"
        )

    # Використовую одну статтю як тестовий запит
    query_idx = 0
    query_vector = embeddings[query_idx]

    print("\nТестова стаття:")
    print(df.iloc[query_idx]["title"])

    # Cosine similarity
    cosine_scores = (
        embeddings @ query_vector
    ) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_vector)
    )

    # Dot product
    dot_scores = embeddings @ query_vector

    # Euclidean distance (L2)
    l2_scores = np.linalg.norm(
        embeddings - query_vector,
        axis=1,
    )

    # Виключаю саму тестову статтю
    cosine_scores[query_idx] = -np.inf
    dot_scores[query_idx] = -np.inf
    l2_scores[query_idx] = np.inf

    # Для cosine і dot більше = краще
    cosine_top = np.argsort(-cosine_scores)[:TOP_K]
    dot_top = np.argsort(-dot_scores)[:TOP_K]

    # Для L2 менше = краще
    l2_top = np.argsort(l2_scores)[:TOP_K]

    print_results(
        "Cosine Similarity",
        cosine_top,
        cosine_scores,
        df,
    )

    print_results(
        "Dot Product",
        dot_top,
        dot_scores,
        df,
    )

    print_results(
        "L2 Distance",
        l2_top,
        l2_scores,
        df,
    )

    results = []

    for metric, indices, scores in [
        ("cosine", cosine_top, cosine_scores),
        ("dot_product", dot_top, dot_scores),
        ("l2", l2_top, l2_scores),
    ]:
        for rank, idx in enumerate(indices, start=1):
            results.append({
                "metric": metric,
                "rank": rank,
                "article_id": df.iloc[idx]["id"],
                "title": df.iloc[idx]["title"],
                "score": float(scores[idx]),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Порівняння ---")

    print(
        "Cosine і Dot Product — однаковий топ-5:",
        set(cosine_top) == set(dot_top),
    )

    print(
        "Cosine і L2 — однаковий топ-5:",
        set(cosine_top) == set(l2_top),
    )

    print(f"\nРезультати збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
