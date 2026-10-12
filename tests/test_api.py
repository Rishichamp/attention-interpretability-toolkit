"""
API tests: the server must start (lifespan loads a model) and every endpoint must answer.

These guard against the model-loading path breaking again: the unit tests in
test_interpretability.py exercise the analysis functions directly and never
start the app.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_source"] in ("toy_untrained_model", "m5p1_trained_checkpoint")


def test_analyse_attention(client):
    r = client.post("/analyse/attention", json={"text": "the cat sat on the mat"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_tokens"] == len("the cat sat on the mat")


def test_probe_rope_is_relative(client):
    r = client.post("/probe/rope", json={})
    assert r.status_code == 200
    dots = r.json()["dot_products"]
    assert max(dots) - min(dots) < 1e-3


@pytest.mark.parametrize("path", ["/probe/alibi", "/probe/sinusoidal"])
def test_probe_get_endpoints(client, path):
    assert client.get(path).status_code == 200


def test_probe_subject_verb(client):
    r = client.post("/probe/subject-verb", json={})
    assert r.status_code == 200
    assert r.json()["probe"] == "subject_verb_agreement"
