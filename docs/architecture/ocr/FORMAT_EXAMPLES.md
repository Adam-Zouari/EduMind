# OCR Output Examples

This file shows representative normalized OCR payloads after `ExtractionResult.to_dict()`.

## PDF

```json
{
  "text": "Annual report content...",
  "source": "annual-report.pdf",
  "format_type": "pdf",
  "num_pages": 12,
  "title": "Annual Report",
  "author": "Finance Team",
  "extractor": "pymupdf",
  "success": true
}
```

## DOCX

```json
{
  "text": "Meeting notes...",
  "source": "notes.docx",
  "format_type": "docx",
  "num_paragraphs": 32,
  "num_tables": 1,
  "author": "Operations",
  "extractor": "python-docx",
  "success": true
}
```

## Image

```json
{
  "text": "Scanned text...",
  "source": "scan.png",
  "format_type": "image",
  "ocr_engine": "tesseract",
  "confidence": 92.5,
  "extractor": "ocr",
  "success": true
}
```

## Audio

```json
{
  "text": "Transcribed lecture...",
  "source": "lecture.mp3",
  "format_type": "audio",
  "language": "en",
  "duration": 1825.5,
  "num_segments": 142,
  "extractor": "whisper",
  "success": true
}
```

## Web

```json
{
  "text": "Article body...",
  "source": "article.html",
  "format_type": "web",
  "title": "Sample Article",
  "author": "Reporter",
  "extractor": "trafilatura",
  "success": true
}
```
