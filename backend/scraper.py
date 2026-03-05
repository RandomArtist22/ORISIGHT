from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from newspaper import Article

from .config import Settings


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class ScrapedDocument:
    source: str
    title: str
    url: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        digest = hashlib.sha1(f"{self.source}:{self.url}:{self.title}".encode("utf-8")).hexdigest()
        return digest


def scrape_oral_cancer_foundation() -> list[ScrapedDocument]:
    url = "https://oralcancerfoundation.org/dental/oral-cancer-images/"
    text, title = _extract_article(url)
    if not text:
        return []
    return [
        ScrapedDocument(
            source="Oral Cancer Foundation",
            title=title or "Oral Cancer Foundation - Oral Cancer Images",
            url=url,
            text=text,
            metadata={"type": "clinical_reference"},
        )
    ]


def scrape_pubmed(query: str = "oral premalignant lesions", max_results: int = 8) -> list[ScrapedDocument]:
    docs: list[ScrapedDocument] = []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        search_resp = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": max_results, "term": query},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        search_resp.raise_for_status()
        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return docs

        fetch_resp = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        fetch_resp.raise_for_status()

        soup = BeautifulSoup(fetch_resp.text, "xml")
        for article in soup.find_all("PubmedArticle"):
            title = (article.find("ArticleTitle").text or "").strip() if article.find("ArticleTitle") else ""
            abstract_parts = [node.get_text(" ", strip=True) for node in article.find_all("AbstractText")]
            abstract = " ".join(part for part in abstract_parts if part)
            pmid_node = article.find("PMID")
            pmid = pmid_node.text.strip() if pmid_node else "unknown"
            if not abstract:
                continue
            docs.append(
                ScrapedDocument(
                    source="PubMed",
                    title=title or f"PubMed {pmid}",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    text=abstract,
                    metadata={"pmid": pmid, "query": query},
                )
            )
    except requests.RequestException:
        return docs

    return docs


def scrape_dataset_metadata() -> list[ScrapedDocument]:
    urls = [
        "https://data.mendeley.com/datasets/mhjyrn35p4",
        "https://data.mendeley.com/datasets/bbmmm4wgr8",
        "https://www.sciencedirect.com/science/article/pii/S1368837524002641",
        "https://www.nature.com/articles/s41597-024-04099-x",
    ]

    docs: list[ScrapedDocument] = []
    for url in urls:
        text, title = _extract_article(url)
        if not text:
            continue
        docs.append(
            ScrapedDocument(
                source="Dataset Metadata",
                title=title or _slugify(url),
                url=url,
                text=text,
                metadata={"type": "dataset_metadata"},
            )
        )
    return docs


def scrape_who_oral_health() -> list[ScrapedDocument]:
    url = "https://www.who.int/news-room/fact-sheets/detail/oral-health"
    text, title = _extract_article(url)
    if not text:
        return []
    return [
        ScrapedDocument(
            source="WHO",
            title=title or "WHO Oral Health",
            url=url,
            text=text,
            metadata={"type": "global_guideline"},
        )
    ]


def scrape_ncbi_oral_disease_summaries() -> list[ScrapedDocument]:
    urls = [
        "https://www.ncbi.nlm.nih.gov/books/NBK65746/",
        "https://www.ncbi.nlm.nih.gov/books/NBK564976/",
    ]
    docs: list[ScrapedDocument] = []
    for url in urls:
        text, title = _extract_article(url)
        if not text:
            continue
        docs.append(
            ScrapedDocument(
                source="NCBI",
                title=title or "NCBI Oral Disease Summary",
                url=url,
                text=text,
                metadata={"type": "ncbi_summary"},
            )
        )
    return docs


def run_scraping_pipeline(settings: Settings, pubmed_query: str, pubmed_max_results: int) -> dict[str, Any]:
    docs: list[ScrapedDocument] = []
    docs.extend(scrape_oral_cancer_foundation())
    docs.extend(scrape_dataset_metadata())
    docs.extend(scrape_who_oral_health())
    docs.extend(scrape_ncbi_oral_disease_summaries())
    docs.extend(scrape_pubmed(pubmed_query, pubmed_max_results))

    saved = save_scraped_documents(docs, Path(settings.raw_docs_dir))
    return {
        "scraped_count": len(docs),
        "saved_count": saved,
        "output_dir": settings.raw_docs_dir,
        "sources": sorted({doc.source for doc in docs}),
    }


def save_scraped_documents(docs: list[ScrapedDocument], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for doc in docs:
        cleaned = clean_text(doc.text)
        if not cleaned:
            continue
        payload = {
            "id": doc.doc_id,
            "source": doc.source,
            "title": doc.title,
            "url": doc.url,
            "text": cleaned,
            "metadata": doc.metadata,
        }
        with (output_dir / f"{doc.doc_id}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        saved += 1

    return saved


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_article(url: str) -> tuple[str, str]:
    html_text = ""
    title = ""

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        html_text = response.text
        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.text or "").strip() if soup.title else ""
        body_text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    except requests.RequestException:
        body_text = ""

    if len(body_text) > 500:
        return clean_text(body_text), title

    try:
        article = Article(url)
        article.download(input_html=html_text if html_text else None)
        article.parse()
        extracted = article.text.strip()
        if extracted:
            return clean_text(extracted), title or article.title
    except Exception:
        pass

    return clean_text(body_text), title


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")
