
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from tqdm import tqdm
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"
OUTPUT_FILE = BASE_DIR / "embeddings" / "embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"

BATCH_SIZE = 8
MAX_LENGTH = 512


def mean_pooling(last_hidden_state, attention_mask):
    """Обчислюю середнє представлення токенів."""
    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1,
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


def normalize_embeddings(embeddings):
    """Нормалізую вектори до одиничної довжини."""
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return embeddings / np.maximum(norms, 1e-12)


def create_embeddings():
    if not DATA_FILE.exists():
        print(f"Не знайдено датасет: {DATA_FILE}")
        return

    df = pd.read_parquet(DATA_FILE)
    print(f"Завантажено статей: {len(df)}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Пристрій для обчислень: {device}")
    print("Завантажую SPECTER2...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoAdapterModel.from_pretrained(
        MODEL_NAME
    )

    adapter_name = model.load_adapter(
        ADAPTER_NAME,
        source="hf",
        set_active=True,
    )

    model.set_active_adapters(adapter_name)
    model.to(device)
    model.eval()

    # Об'єдную назву та анотацію статті
    # через спеціальний токен [SEP]
    texts = (
        df["title"].fillna("")
        + " "
        + tokenizer.sep_token
        + " "
        + df["abstract"].fillna("")
    ).tolist()

    embeddings = []

    print("Створюю ембеддинги...")

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE)
    ):
        batch = texts[
            start:start + BATCH_SIZE
        ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            batch_embeddings = mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    # Об'єдную всі ембеддинги в один масив
    embeddings = np.vstack(
        embeddings
    ).astype(np.float32)

    # Нормалізую ембеддинги
    embeddings = normalize_embeddings(
        embeddings
    )

    # Перевіряю результат нормалізації
    first_vector_norm = np.linalg.norm(
        embeddings[0]
    )

    average_norm = np.linalg.norm(
        embeddings,
        axis=1,
    ).mean()

    # Створюю папку, якщо її ще немає
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Зберігаю готові ембеддинги
    np.save(
        OUTPUT_FILE,
        embeddings,
    )

    print("\nЕмбеддинги успішно створено!")
    print(f"Кількість статей: {len(df)}")
    print(f"Форма масиву: {embeddings.shape}")
    print(f"Розмірність: {embeddings.shape[1]}")
    print(
        f"Норма першого вектора: "
        f"{first_vector_norm:.6f}"
    )
    print(
        f"Середня норма: "
        f"{average_norm:.6f}"
    )
    print(f"Файл збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_embeddings()
