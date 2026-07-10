"""Web content extraction using Trafilatura and newspaper3k."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from newspaper import Article
import requests
import trafilatura

from ..config import USER_AGENT, WEB_TIMEOUT
from ..core.base_extractor import BaseExtractor, ExtractionResult


class WebExtractor(BaseExtractor):
    """Extract text from HTML files or remote web pages."""

    def __init__(self) -> None:
        super().__init__()

    def extract(
        self,
        file_path: Path | None,
        url: str | None = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract text from an HTML file or a URL."""
        start_time = time.time()
        target = file_path if file_path else url
        self.logger.info(f"Extracting web content: {target}")

        try:
            if file_path and file_path.exists():
                html_content = file_path.read_text(encoding="utf-8", errors="ignore")
            elif url:
                html_content = self._fetch_remote_html(url)
            else:
                raise ValueError("Either file_path or url must be provided")

            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
            )

            metadata = trafilatura.extract_metadata(html_content)
            meta_dict = {
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "sitename": metadata.sitename if metadata else "",
                "extractor": "trafilatura",
            }

            if not text or len(text) < 100:
                self.logger.info("Trying newspaper3k for extraction")
                text, news_meta = self._extract_with_newspaper(html_content, url)
                meta_dict.update(news_meta)

            return ExtractionResult(
                text=text or "",
                metadata=meta_dict,
                format_type="web",
                file_path=str(file_path) if file_path else (url or ""),
                extraction_time=time.time() - start_time,
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"Web extraction failed: {exc}")
            return ExtractionResult(
                text="",
                metadata={},
                format_type="web",
                file_path=str(file_path) if file_path else (url or ""),
                success=False,
                error=str(exc),
            )

    def _extract_with_newspaper(
        self,
        html_content: str,
        url: str | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Extract article text using newspaper3k."""
        article = Article(url or "", language="en")
        article.set_html(html_content)
        article.parse()

        metadata = {
            "title": article.title,
            "authors": ", ".join(article.authors),
            "publish_date": str(article.publish_date) if article.publish_date else "",
            "top_image": article.top_image,
            "extractor": "newspaper3k",
        }

        return article.text, metadata

    def _fetch_remote_html(self, url: str) -> str:
        """Fetch remote HTML using configured timeout and user-agent settings."""
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_TIMEOUT,
        )
        response.raise_for_status()
        return response.text
