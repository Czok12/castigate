import requests

BASE_URL = "http://localhost:8000/api/v1"


def test_create_and_get_book():
    # Buch anlegen
    payload = {
        "titel": "Testbuch",
        "autor": "Max Mustermann",
        "jahr": 2024,
        "verlag": "Testverlag",
        "isbn": "1234567890",
        "rechtsgebiet": "Zivilrecht",
        "auflage": "1. Aufl.",
        "dateiname": "testbuch.pdf",
    }
    r = requests.post(f"{BASE_URL}/books", json=payload)
    assert r.status_code in (200, 201), r.text
    book = r.json()
    assert book["titel"] == payload["titel"]
    book_id = book["id"]

    # Buch abrufen
    r = requests.get(f"{BASE_URL}/books/{book_id}")
    assert r.status_code == 200
    book2 = r.json()
    assert book2["titel"] == payload["titel"]

    # Buch löschen
    r = requests.delete(f"{BASE_URL}/books/{book_id}")
    assert r.status_code in (200, 204)

    # Nach dem Löschen nicht mehr auffindbar
    r = requests.get(f"{BASE_URL}/books/{book_id}")
    assert r.status_code == 404


if __name__ == "__main__":
    test_create_and_get_book()
    print("Alle Bibliotheks-API-Tests erfolgreich!")
