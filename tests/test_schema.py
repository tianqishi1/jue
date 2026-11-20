import pytest
from jue.schema import (SchemaError, format_answer, label_index, question_options,
                        validate_question, validate_questions)

CHOICE = {"type": "choice", "instructions": "转给哪个组？",
          "criteria": {"refund": "退款", "logistics": "物流"}}
NOUL = {"type": "noul", "instructions": "是否要求退款？"}
SCORE = {"type": "score", "instructions": "严重程度？", "criteria": ["轻", "中", "重"]}


def test_validate_ok():
    for q in (CHOICE, NOUL, SCORE):
        assert validate_question(q)["type"] == q["type"]


def test_validate_bad():
    with pytest.raises(SchemaError):
        validate_question({"type": "essay", "instructions": "x"})
    with pytest.raises(SchemaError):
        validate_question({"type": "choice", "instructions": "x", "criteria": {"only": "one"}})
    with pytest.raises(SchemaError):
        validate_question({"type": "score", "instructions": "x", "criteria": "轻中重"})
    with pytest.raises(SchemaError):
        validate_questions({})


def test_options_order():
    assert [k for k, _ in question_options(CHOICE)] == ["refund", "logistics"]
    assert [k for k, _ in question_options(NOUL)] == ["true", "false"]
    assert [k for k, _ in question_options(SCORE)] == ["0", "1", "2"]


def test_label_index():
    assert label_index(CHOICE, "logistics") == 1
    assert label_index(NOUL, True) == 0
    assert label_index(NOUL, "否") == 1
    assert label_index(SCORE, "重") == 2
    assert label_index(SCORE, 1) == 1
    with pytest.raises(SchemaError):
        label_index(CHOICE, "nonexistent")


def test_format_answer():
    a = format_answer(CHOICE, [0.9, 0.1])
    assert a["answer"] == "refund" and a["probs"]["refund"] == 0.9 and not a["abstained"]
    b = format_answer(NOUL, [0.2, 0.8])
    assert b["answer"] is False and b["prob"] == 0.2
    c = format_answer(SCORE, [0.1, 0.2, 0.7])
    assert c["answer"] == "重" and abs(c["expected"] - 1.6) < 1e-6
    d = format_answer(CHOICE, [0.4, 0.6], abstain_threshold=0.9)
    assert d["abstained"] and d["answer"] is None
