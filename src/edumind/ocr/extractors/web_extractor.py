"""Web content extraction using Trafilatura and newspaper3k."""

from __future__ import annotations

import time
from pathlib import Path

import requests
import trafilatura
from newspaper import Article

from ..config import USER_AGENT, WEB_TIMEOUT
from ..core.base_extractor import BaseExtractor, ExtractionResult


class WebExtractor(BaseExtractor):
    """Extract text from HTML files or remote web pages."""

    def extract(
        self,
        file_path: Path,
        **kwargs: object,
    ) -> ExtractionResult:
        """Extract text from an HTML file or a URL."""
        url = kwargs.get("url")
        resolved_url = url if isinstance(url, str) else None
        start_time = time.time()
        target = file_path if file_path else resolved_url
        self.logger.info(f"Extracting web content: {target}")

        try:
            html_content = self._load_html_content(file_path=file_path, url=resolved_url)
            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
            )
            metadata = self._build_trafilatura_metadata(html_content)

            if not text or len(text) < 100:
                self.logger.info("Trying newspaper3k for extraction")
                text, fallback_metadata = self._extract_with_newspaper(html_content, resolved_url)
                metadata.update(fallback_metadata)

            return ExtractionResult(
                text=text or "",
                metadata=metadata,
                format_type="web",
                file_path=str(file_path) if file_path else (resolved_url or ""),
                extraction_time=time.time() - start_time,
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"Web extraction failed: {exc}")
            return ExtractionResult(
                text="",
                metadata={},
                format_type="web",
                file_path=str(file_path) if file_path else (resolved_url or ""),
                success=False,
                error=str(exc),
            )

    def _load_html_content(self, *, file_path: Path, url: str | None) -> str:
        """Load HTML from a local file or a remote URL."""
        if file_path.exists():
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if url:
            return self._fetch_remote_html(url)
        raise ValueError("Either file_path or url must be provided")

    def _build_trafilatura_metadata(self, html_content: str) -> dict[str, object]:
        """Build normalized metadata from Trafilatura extraction."""
        metadata = trafilatura.extract_metadata(html_content)
        return {
            "title": metadata.title if metadata else "",
            "author": metadata.author if metadata else "",
            "date": metadata.date if metadata else "",
            "sitename": metadata.sitename if metadata else "",
            "extractor": "trafilatura",
        }

    def _extract_with_newspaper(
        self,
        html_content: str,
        url: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Extract article text using newspaper3k."""
        article = Article(url or "", language="en")
        article.set_html(html_content)
        article.parse()

        return article.text, {
            "title": article.title,
            "authors": ", ".join(article.authors),
            "publish_date": str(article.publish_date) if article.publish_date else "",
            "top_image": article.top_image,
            "extractor": "newspaper3k",
        }

    def _fetch_remote_html(self, url: str) -> str:
        """Fetch remote HTML using configured timeout and user-agent settings."""
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_TIMEOUT,
        )
        response.raise_for_status()
        return response.text
