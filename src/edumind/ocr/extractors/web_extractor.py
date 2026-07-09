"""
Web content extraction using Trafilatura and newspaper3k
"""
import trafilatura
from newspaper import Article
from pathlib import Path
from typing import Dict
import time
from ..core.base_extractor import BaseExtractor, ExtractionResult
from ..config import WEB_TIMEOUT, USER_AGENT

class WebExtractor(BaseExtractor):
    """Extract text from HTML/web content"""
    
    def __init__(self):
        super().__init__()
    
    def extract(self, file_path: Path, url: str = None, **kwargs) -> ExtractionResult:
        """Extract text from HTML file or URL"""
        start_time = time.time()
        self.logger.info(f"Extracting web content: {file_path if file_path else url}")
        
        try:
            if file_path and file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
            elif url:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    raise ValueError(f"Could not download URL: {url}")
                html_content = downloaded
            else:
                raise ValueError("Either file_path or url must be provided")
            
            # Try Trafilatura first (better for articles)
            text = trafilatura.extract(html_content, include_comments=False, include_tables=True)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html_content)
            meta_dict = {
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "sitename": metadata.sitename if metadata else "",
                "extractor": "trafilatura"
            }
            
            # If Trafilatura fails, try newspaper3k
            if not text or len(text) < 100:
                self.logger.info("Trying newspaper3k for extraction")
                text, news_meta = self._extract_with_newspaper(html_content, url)
                meta_dict.update(news_meta)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                text=text or "",
                metadata=meta_dict,
                format_type="web",
                file_path=str(file_path) if file_path else url,
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Web extraction failed: {e}")
            return self._create_error_result(file_path or Path(url), str(e))
    
    def _extract_with_newspaper(self, html_content: str, url: str = None) -> tuple[str, Dict]:
        """Extract using newspaper3k"""
        article = Article(url or "", language='en')
        article.set_html(html_content)
        article.parse()
        
        metadata = {
            "title": article.title,
            "authors": ", ".join(article.authors),
            "publish_date": str(article.publish_date) if article.publish_date else "",
            "top_image": article.top_image,
            "extractor": "newspaper3k"
        }
        
        return article.text, metadata