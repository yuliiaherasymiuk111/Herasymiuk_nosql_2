
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = BASE_DIR / "data" / "arxiv_subset.parquet"

TARGET_COUNT = 8000
START_YEAR = 2007
END_YEAR = 2025

YEARS = list(range(START_YEAR, END_YEAR + 1))
BASE_QUOTA = TARGET_COUNT // len(YEARS)
EXTRA = TARGET_COUNT % len(YEARS)

YEAR_QUOTAS = {
    year: BASE_QUOTA + (1 if i < EXTRA else 0)
    for i, year in enumerate(YEARS)
}


def clean_text(value):
    """Очищаю текст від зайвих пробілів."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def get_year(article):
    """Визначаю рік за датою першої версії статті."""
    versions = article.get("versions", [])

    if versions:
        created = versions[0].get("created", "")
        match = re.search(r"\b(?:19|20)\d{2}\b", created)

        if match:
            return int(match.group())

    return None


def prepare_data():
    if not INPUT_FILE.exists():
        print(f"Файл не знайдено: {INPUT_FILE}")
        return

    articles_by_year = defaultdict(list)
    total_selected = 0

    print("Починаю формувати вибірку за різними роками...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            year = get_year(article)

            if year not in YEAR_QUOTAS:
                continue

            if len(articles_by_year[year]) >= YEAR_QUOTAS[year]:
                continue

            article_id = clean_text(article.get("id"))
            title = clean_text(article.get("title"))
            abstract = clean_text(article.get("abstract"))
            authors = clean_text(article.get("authors"))
            categories = clean_text(article.get("categories"))

            if not all([article_id, title, abstract, categories]):
                continue

            primary_category = categories.split()[0]

            articles_by_year[year].append(
                {
                    "id": article_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "category": primary_category,
                    "categories": categories,
                    "text": f"{title}. {abstract}",
                }
            )

            total_selected += 1

            if total_selected >= TARGET_COUNT:
                break

    articles = [
        article
        for year in YEARS
        for article in articles_by_year[year]
    ]

    if not articles:
        print("Не вдалося знайти статті.")
        return

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["id"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nПідготовку завершено!")
    print(f"Кількість статей: {len(df)}")
    print(f"Роки: {df['year'].min()}–{df['year'].max()}")
    print(f"Кількість категорій: {df['category'].nunique()}")

    print("\nРозподіл статей за роками:")
    print(df["year"].value_counts().sort_index().to_string())

    print(f"\nФайл збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_data()
