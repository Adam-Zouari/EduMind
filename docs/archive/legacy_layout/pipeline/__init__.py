"""
Pipeline Module - Orchestrates OCR and RAG

Note: This module contains microservices. Import them individually:
- from pipeline.ocr_service import app as ocr_app
- from pipeline.rag_service import app as rag_app
- from pipeline.orchestrator_api import APIOrchestrator
"""

# Don't import anything by default to avoid import errors
# Each service should be run independently

__all__ = []

