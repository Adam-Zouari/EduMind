# Extraction architecture

The extraction path is `source -> classifier -> router -> extractor -> normalization -> ExtractedDocument`. Its public contracts, cache identity, supported modalities, and benchmark boundary are documented in [`src/edumind/extraction/doc.md`](../../../src/edumind/extraction/doc.md).

The rename from OCR to extraction is intentional and has no compatibility shim: OCR is one possible image or scanned-page technique, not the subsystem. Web, structured tables, formulas, and dedicated forms are outside the current scope.
