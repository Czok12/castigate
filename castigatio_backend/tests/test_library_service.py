"""Unit-Tests für LibraryService (Bibliotheksverwaltung).
Alle externen Abhängigkeiten werden gemockt.
"""

import tempfile

import pytest

from app.models.library import BookMetadataCreate
from app.services.library_service import LibraryService


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        yield tf.name


@pytest.fixture
def service(temp_db_path):
    return LibraryService(db_path=temp_db_path)


def test_add_and_get_book(service):
    book = BookMetadataCreate(
        titel="BGB AT",
        autor="Medicus",
        auflage="11. Aufl.",
        jahr=2022,
        verlag="Vahlen",
        isbn="978-3-8006-7000-0",
        rechtsgebiet="Zivilrecht",
        dateiname="Medicus_BGB_AT.pdf",
    )
    added = service.add_book(book)
    assert added.titel == "BGB AT"
    assert added.autor == "Medicus"
    assert added.jahr == 2022
    # ID-Format prüfen
    assert added.id.startswith("medicus_bgb_at_2022")
    # Buch abrufen
    fetched = service.get_book_by_id(added.id)
    assert fetched is not None
    assert fetched.titel == "BGB AT"


def test_add_duplicate_book_raises(service):
    book = BookMetadataCreate(
        titel="BGB AT",
        autor="Medicus",
        jahr=2022,
        rechtsgebiet="Zivilrecht",
    )
    service.add_book(book)
    with pytest.raises(ValueError):
        service.add_book(book)


def test_delete_book(service):
    book = BookMetadataCreate(
        titel="BGB AT",
        autor="Medicus",
        jahr=2022,
        rechtsgebiet="Zivilrecht",
    )
    added = service.add_book(book)
    assert service.delete_book(added.id) is True
    assert service.get_book_by_id(added.id) is None


def test_get_all_books(service):
    book1 = BookMetadataCreate(
        titel="BGB AT",
        autor="Medicus",
        jahr=2022,
        rechtsgebiet="Zivilrecht",
    )
    book2 = BookMetadataCreate(
        titel="StPO",
        autor="Meyer-Goßner",
        jahr=2021,
        rechtsgebiet="Strafrecht",
    )
    service.add_book(book1)
    service.add_book(book2)
    all_books = service.get_all_books()
    assert len(all_books) == 2
    ids = [b.id for b in all_books]
    assert any("medicus_bgb_at_2022" in i for i in ids)
    assert any(
        "meyer_goßner_stpo_2021" in i or "meyer_go__ner_stpo_2021" in i for i in ids
    )


def test_delete_book_removes_faiss_index(service, monkeypatch, tmp_path):
    """Testet, dass beim Löschen eines Buches auch das FAISS-Index-Verzeichnis entfernt wird."""
    import shutil

    from app.core import config

    # Buch anlegen
    book = BookMetadataCreate(
        titel="BGB AT",
        autor="Medicus",
        jahr=2022,
        rechtsgebiet="Zivilrecht",
    )
    added = service.add_book(book)
    book_id = added.id

    # Simuliere FAISS-Index-Verzeichnis
    faiss_dir = tmp_path / book_id
    faiss_dir.mkdir()
    (faiss_dir / "dummy.index").write_text("test")

    # Patch FAISS_DB_PATH auf tmp_path
    monkeypatch.setattr(config, "FAISS_DB_PATH", tmp_path)

    # Patch shutil.rmtree, um Aufruf zu überwachen
    called = {}

    def fake_rmtree(path):
        called["path"] = path
        shutil.rmtree.__wrapped__(path)

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    # Löschen
    result = service.delete_book(book_id)
    assert result is True
    assert called["path"] == faiss_dir
    assert not faiss_dir.exists()
