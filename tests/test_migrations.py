"""The migration chain works on a fresh database AND on a populated prototype database."""

import sqlite3

from conftest import ROOT


def test_upgrade_populated_prototype_database(tmp_path, monkeypatch):
    monkeypatch.delenv("MOVEMATCH_ENV", raising=False)
    from flask_migrate import downgrade, upgrade

    from app import create_app
    db_file = tmp_path / "proto.sqlite3"
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file}",
                      "STORAGE_DIR": tmp_path / "s", "SECRET_KEY": "x" * 40})
    migrations = str(ROOT / "migrations")
    with app.app_context():
        upgrade(directory=migrations, revision="b57605eb2ee6")            # the prototype's schema
    con = sqlite3.connect(db_file)
    con.execute("INSERT INTO video (id, original_name, stored_filename, annotated_filename, status, created_at) "
                "VALUES (1, 'dance.mp4', 'abc.mp4', 'abc_annotated.mp4', 'completed', '2026-10-12 10:00:00')")
    con.execute("INSERT INTO attempt (id, video_id, recording_filename, speed, status, details_json, created_at) "
                "VALUES (1, 1, 'r.webm', 0.5, 'recorded', '{}', '2026-10-12 10:05:00')")
    con.execute("INSERT INTO evaluation (video_id, speed, score) VALUES (1, 0.5, 80)")
    con.commit()
    con.close()

    with app.app_context():
        upgrade(directory=migrations)
    con = sqlite3.connect(db_file)
    assert con.execute("SELECT status, playback_filename FROM video").fetchone() == ("needs_reprocessing", None)
    assert con.execute("SELECT status, view_mirrored, follow_mode FROM attempt").fetchone() == ("uploaded", 1, "same_side")
    assert con.execute("SELECT count(*) FROM evaluation").fetchone() == (0,)
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    con.close()

    with app.app_context():                                                 # and back down again
        downgrade(directory=migrations, revision="b57605eb2ee6")
    con = sqlite3.connect(db_file)
    assert con.execute("SELECT count(*) FROM attempt").fetchone() == (1,)
    con.close()
