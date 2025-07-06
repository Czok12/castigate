"""
API-Tests für die Bibliotheks-Endpunkte.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def example_book():
    return {
        "titel": "BGB AT",
        "autor": "Medicus",
        "auflage": "11. Aufl.",
        "jahr": 2022,
        "verlag": "Vahlen",
        "isbn": "978-3-8006-7000-0",
        "rechtsgebiet": "Zivilrecht",
        "dateiname": "Medicus_BGB_AT.pdf",
    }


def test_create_and_get_book(example_book):
    # Buch anlegen
    response = client.post("/api/v1/books", json=example_book)
    assert response.status_code == 201
    data = response.json()
    assert data["titel"] == example_book["titel"]
    book_id = data["id"]
    # Buch abrufen
    get_resp = client.get(f"/api/v1/books/{book_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["titel"] == example_book["titel"]
    # Alle Bücher abrufen
    all_resp = client.get("/api/v1/books")
    assert all_resp.status_code == 200
    assert any(b["id"] == book_id for b in all_resp.json())
    # Buch löschen
    del_resp = client.delete(f"/api/v1/books/{book_id}")
    assert del_resp.status_code == 204
    # Nach dem Löschen nicht mehr auffindbar
    get_resp2 = client.get(f"/api/v1/books/{book_id}")
    assert get_resp2.status_code == 404


def test_duplicate_book(example_book):
    # Buch anlegen
    response = client.post("/api/v1/books", json=example_book)
    assert response.status_code == 201
    # Nochmals anlegen (sollte Fehler geben)
    response2 = client.post("/api/v1/books", json=example_book)
    assert response2.status_code == 409
    # Aufräumen
    book_id = response.json()["id"]
    client.delete(f"/api/v1/books/{book_id}")
