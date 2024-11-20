from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
client = TestClient(app)


def test_get_scenarios_all() -> None:
    response = client.get("/scenarios/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
