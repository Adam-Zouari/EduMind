"""Observed component identities, separate from the Docling package version.

HF commits come from the first line of each prepared component's
.cache/huggingface/download/config.json.metadata, inspected 2026-11-13.
File SHA256 values were measured from the existing project-owned RapidOcr and
EasyOcr files on that date. They freeze observed bytes, not quality claims.
Preparation verifies fresh downloads against these identities, never `main`.
"""

HF_COMPONENTS = {
    "layout": (
        "docling-project/docling-layout-heron",
        "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    ),
    "tableformer": (
        "docling-project/docling-models",
        "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
    ),
    "code_formula": (
        "docling-project/CodeFormulaV2",
        "ecedbe111d15c2dc60bfd4a823cbe80127b58af4",
    ),
}

FILE_COMPONENTS = {
    "rapidocr": {
        "folder": "RapidOcr",
        "source": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/",
        "files": {
            "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
            "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
            "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
            "resources/fonts/FZYTK.TTF": "4065a23df6823c8e2b69a0e76d02f02a6470b8774a5e91086609701ad95cc33f",
        },
    },
    "easyocr": {
        "folder": "EasyOcr",
        "source": "https://github.com/JaidedAI/EasyOCR/releases/",
        "files": {
            "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
            "english_g2.pth": "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8",
        },
    },
}
