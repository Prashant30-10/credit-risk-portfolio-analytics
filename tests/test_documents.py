import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_pdfs_are_exactly_one_page():
    for name in ["architecture.pdf", "solution_overview.pdf"]:
        data = (ROOT / "docs" / name).read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
        assert pages == 1, f"{name} has {pages} pages"

