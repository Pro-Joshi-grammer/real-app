"""Self-check for answer extraction. Run: python backend/test_extract.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.main import extract_answer

cases = [
    # tagged, with CoT prefix/suffix outside the tags
    ("Okay the user is asking me to analyze this question. "
     "<answer>C) Large Language Model</answer> "
     "hope that helps!",
     "C) Large Language Model"),
    # tags with arbitrary content (numbers/symbols)
    ("<ANSWER>sqrt(144) = 12</ANSWER>", "sqrt(144) = 12"),
    # no tags -> normalizer path
    ("The correct answer is 42.", "42."),
    # empty
    ("", ""),
]

for raw, want in cases:
    got = extract_answer(raw)
    assert got == want, f"FAIL: {raw!r}\n  got      {got!r}\n  expected {want!r}"

print(f"OK — {len(cases)} cases passed")
