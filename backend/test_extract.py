"""Self-check for answer extraction. Run: python backend/test_extract.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.main import extract_answer

cases = [
    # structured-output JSON (schema mode)
    ('{"answer": "C) Large Language Model"}', "C) Large Language Model"),
    # JSON wrapped in markdown fences (json_object mode)
    ('```json\n{"answer": "sqrt(144) = 12"}\n```', "sqrt(144) = 12"),
    # multiline answer preserved (code / formulas)
    ('{ "answer": "x = 1\\ny = 2" }', "x = 1\ny = 2"),
    # JSON without an answer key -> invalid/empty
    ('{"foo": "bar"}', ""),
    # non-object JSON -> invalid/empty
    ('["not", "an", "object"]', ""),
    # legacy <answer>…</answer> contract still accepted (Bedrock fallback)
    ("<answer>C) Large Language Model</answer>", "C) Large Language Model"),
    # untagged raw reasoning -> invalid/empty (never shown)
    ("The correct answer is 42.", ""),
    # empty
    ("", ""),
]

for raw, want in cases:
    got = extract_answer(raw)
    assert got == want, f"FAIL: {raw!r}\n  got      {got!r}\n  expected {want!r}"

print(f"OK — {len(cases)} cases passed")
