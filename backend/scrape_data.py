from __future__ import annotations

import argparse
import json

try:
    from .config import get_settings
    from .scraper import run_scraping_pipeline
except ImportError:  # pragma: no cover
    from backend.config import get_settings
    from backend.scraper import run_scraping_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape oral lesion references for ORISIGHT RAG")
    parser.add_argument("--query", default="oral potentially malignant disorders", help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=8, help="Max PubMed abstracts")
    args = parser.parse_args()

    settings = get_settings()
    stats = run_scraping_pipeline(settings, args.query, args.max_results)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
