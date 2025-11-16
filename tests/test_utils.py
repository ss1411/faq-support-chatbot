import re
from src.utils import sentence_split

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def test_sentence_split():
    text = (
        "First sentence. Second sentence!\n\n"
        "Third sentence? Fourth sentence follows.   "
    )
    sentences = sentence_split(text)
    assert isinstance(sentences, list)
    assert len(sentences) >= 4
    assert all(isinstance(s, str) and s.strip() for s in sentences)
    # ensure original content is preserved when joined
    joined = " ".join(sentences)
    assert _norm_ws("First sentence. Second sentence! Third sentence? Fourth sentence follows.") in _norm_ws(joined)


# run a single test function by node id
# python -m pytest -q tests/test_chunking.py::test_sentence_split --verbose
