"""server 回归测试：重点防 FastAPI 延迟注解导致 Body 模型被当 query 参数的坑。"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from jue.backends import make_backend
from jue.server import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(make_backend("mock")))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_decisions_body(client):
    r = client.post("/v1/decisions", json={
        "state": "测试",
        "questions": {
            "q": {"type": "noul", "instructions": "是否？"},
            "c": {"type": "choice", "instructions": "选哪个？", "criteria": {"x": "X", "y": "Y"}},
        },
    })
    assert r.status_code == 200, r.text  # 回归：曾被解析为 query 参数而 422
    answers = r.json()["answers"]
    assert isinstance(answers["q"]["answer"], bool)
    assert abs(sum(answers["c"]["probs"].values()) - 1) < 1e-5


def test_decisions_invalid_schema(client):
    r = client.post("/v1/decisions", json={"state": "x", "questions": {"q": {"type": "essay"}}})
    assert r.status_code == 422


def test_decisions_abstain(client):
    r = client.post("/v1/decisions", json={
        "state": "测试", "abstain_threshold": 0.999,
        "questions": {"q": {"type": "noul", "instructions": "是否？"}},
    })
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["abstained"] is True
