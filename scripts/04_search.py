
import os
from pathlib import Path

import torch
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "arxiv-semantic-search",
)


def load_model():
    """Завантажую модель SPECTER2 для пошуку."""
    print("Завантажую модель SPECTER2...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    adapter = model.load_adapter(
        ADAPTER_NAME,
        source="hf",
    )

    model.set_active_adapters(adapter)
    model.eval()

    return tokenizer, model


def create_query_embedding(query, tokenizer, model):
    """Перетворюю пошуковий запит на вектор."""
    inputs = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1)
        mask = mask.expand(outputs.last_hidden_state.size()).float()

        embedding = (
            (outputs.last_hidden_state * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

    return embedding[0].cpu().numpy().tolist()


def semantic_search(
    query,
    tokenizer,
    model,
    index,
    top_k=5,
    year=None,
    category=None,
    year_before=None,
    year_from=None,
):
    """Шукаю наукові статті за змістом із фільтрами."""
    vector = create_query_embedding(query, tokenizer, model)

    filters = {}

    if year is not None:
        filters["year"] = {"$eq": int(year)}
    elif year_before is not None or year_from is not None:
        year_filter = {}

        if year_before is not None:
            year_filter["$lt"] = int(year_before)

        if year_from is not None:
            year_filter["$gte"] = int(year_from)

        filters["year"] = year_filter

    if category:
        filters["category"] = {"$eq": category}

    search_params = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if filters:
        search_params["filter"] = filters

    results = index.query(**search_params)

    print(f"\nПошуковий запит: {query}")

    if filters:
        print(f"Фільтри: {filters}")

    print("\nРезультати пошуку:")

    if not results.matches:
        print("Статей не знайдено.")
        return

    for number, match in enumerate(results.matches, start=1):
        metadata = match.metadata or {}

        print(f"\n{number}. {metadata.get('title', 'Без назви')}")
        print(f"   ID: {match.id}")
        print(f"   Рік: {metadata.get('year', '-')}")
        print(f"   Категорія: {metadata.get('category', '-')}")
        print(f"   Similarity score: {match.score:.4f}")
        print(f"   Автори: {metadata.get('authors', '-')}")


def main():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        print("Помилка: PINECONE_API_KEY не знайдено.")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    tokenizer, model = load_model()

    query = "machine learning methods for image classification"

    # 1. Пошук без фільтрів
    semantic_search(
        query=query,
        tokenizer=tokenizer,
        model=model,
        index=index,
        top_k=5,
    )

    # 2. Пошук за конкретним роком
    semantic_search(
        query=query,
        tokenizer=tokenizer,
        model=model,
        index=index,
        top_k=5,
        year=2020,
    )

    # 3. Пошук за категорією
    semantic_search(
        query=query,
        tokenizer=tokenizer,
        model=model,
        index=index,
        top_k=5,
        category="cs.CV",
    )

    # 4. Новіші статті з Machine Learning
    semantic_search(
        query=query,
        tokenizer=tokenizer,
        model=model,
        index=index,
        top_k=5,
        category="cs.LG",
        year_from=2020,
    )

    # 5. Статті, опубліковані до 2015 року
    semantic_search(
        query=query,
        tokenizer=tokenizer,
        model=model,
        index=index,
        top_k=5,
        year_before=2015,
    )


if __name__ == "__main__":
    main()
