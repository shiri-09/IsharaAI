from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# columns added after the first release; create_all() does not alter existing tables, so add them here
_ADDED_COLUMNS = {
    "conversations": [("context", "VARCHAR(24) DEFAULT 'general'"), ("mode", "VARCHAR(8) DEFAULT 'ai'")],
    "messages": [("interpreted", "TEXT DEFAULT ''"), ("meta", "TEXT DEFAULT '{}'")],
    "user_prefs": [
        ("text_size", "VARCHAR(8) DEFAULT 'normal'"),
        ("scan_ms", "INTEGER DEFAULT 1600"),
        ("avatar_speed", "VARCHAR(8) DEFAULT 'normal'"),
        ("auto_speak", "INTEGER DEFAULT 1"),
        ("high_contrast", "INTEGER DEFAULT 0"),
    ],
}


def migrate() -> None:
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
