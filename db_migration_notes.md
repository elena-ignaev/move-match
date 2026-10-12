# Step-by-step db update with flask-migrate
0. Back up and install
```
cp videos.sqlite3 videos.backup.sqlite3
pip install flask-migrate        # also add it to requirements.txt
```

1. Wire it up

`extensions.py`:
```
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()
```

`app.py`: change the import, allow a database override (needed in step 3), and initialize it:
```
python
from extensions import db, migrate

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'videos.sqlite3').as_posix()}"
)

db.init_app(app)
migrate.init_app(app, db)
```

2. Create the migrations folder

```
flask db init
```

3. Generate a baseline from an empty scratch database. The baseline has to represent your current schema, so do this before adding the new column to models.py:

```
DATABASE_URL=sqlite:////tmp/scratch.sqlite3 flask db migrate -m "baseline schema"
rm /tmp/scratch.sqlite3
```

You should see Detected added table for video, attempt and evaluation. 

4. Mark your real database as already at the baseline

```
flask db stamp head
```

This only records a version number and changes no tables. Skipping it would make upgrade try to create tables that already exist.

5. Add the column to `models.py`
```
annotated_filename = db.Column(db.String(255), nullable=True)
landmarks_path = db.Column(db.String(255), nullable=True)
```

6. Generate the migration and read it before applying

```
flask db migrate -m "add landmarks_path to video"
```

Open the new file in `migrations/versions/`. It should contain only this:

```
with op.batch_alter_table('video', schema=None) as batch_op:
    batch_op.add_column(sa.Column('landmarks_path', sa.String(length=255), nullable=True))
```

If it contains anything else, it means your real schema differs from the models.

7. Apply it

```
flask db upgrade
```

Existing rows get `NULL` for `landmarks_path`. Flask-Migrate 4.x uses SQLite-safe batch mode by default, so later column changes will work too.

# Use the column in the upload handler

After `analyze_video` returns, store the path relative to `storage/` so the project can move without breaking it:

```
video.landmarks_path = Path(metadata["landmarks_path"]).relative_to(STORAGE_DIR).as_posix()
```

That stores `landmarks/<stem>.npz`. To read it later, use `STORAGE_DIR / video.landmarks_path`. Videos uploaded before this change stay `NULL` until the backfill step.

# Going forward

Edit `models.py`, then run `flask db migrate -m "..."`, review the generated file, and run `flask db upgrade`. Commit `migrations/` to git. A fresh clone needs `flask db upgrade` once, since `create_all()` no longer builds the tables.