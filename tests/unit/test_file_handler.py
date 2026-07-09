from pathlib import Path

from edumind.ocr.utils.file_handler import FileHandler


def test_file_handler_hash_and_size(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("edumind", encoding="utf-8")

    assert FileHandler.validate_file(sample) is True
    assert FileHandler.get_file_size(sample) == len("edumind")
    assert len(FileHandler.get_file_hash(sample)) == 32


def test_file_handler_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert FileHandler.validate_file(missing) is False
