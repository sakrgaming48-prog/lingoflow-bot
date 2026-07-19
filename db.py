"""
LingoFlow — Async Database Layer
=================================
Provides all database operations using aiosqlite.
Each function opens its own connection to avoid SQLite locking
issues in an async context.

Multi-Deck Architecture
-----------------------
Each user has an `active_deck` field.  All cart operations are
scoped to that deck, allowing identical terms in different decks.

Dynamic API Keys
----------------
Users can store a personal `gemini_key` and `selected_model`
that override the server defaults at generation time.
"""

import aiosqlite
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Default model shown to users when none is set ─────────────
DEFAULT_MODEL = "models/gemini-2.5-flash"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Schema Init & Migration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def init_db(db_path: str) -> None:
    """Initialize and safely migrate the database schema.

    All ALTER TABLE calls are wrapped in try/except so they are
    silently skipped on databases that already have the column.
    No data is ever dropped.
    """
    logger.info("Initializing database at: %s", db_path)

    async with aiosqlite.connect(db_path) as db:

        # ── users table ───────────────────────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id          INTEGER PRIMARY KEY,
                main_deck        TEXT,
                last_activity    TIMESTAMP,
                total_exported   INTEGER DEFAULT 0,
                gemini_key       TEXT,
                selected_model   TEXT DEFAULT 'models/gemini-2.5-flash',
                active_deck      TEXT DEFAULT 'Default'
            )
        """)

        # Backward-compatible migrations for existing databases
        _safe_alters = [
            "ALTER TABLE users ADD COLUMN total_exported INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN gemini_key TEXT",
            f"ALTER TABLE users ADD COLUMN selected_model TEXT DEFAULT '{DEFAULT_MODEL}'",
            "ALTER TABLE users ADD COLUMN active_deck TEXT DEFAULT 'Default'",
        ]
        for stmt in _safe_alters:
            try:
                await db.execute(stmt)
            except aiosqlite.OperationalError:
                pass  # column already exists

        # ── cart table ────────────────────────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cart (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER,
                deck_name      TEXT NOT NULL DEFAULT 'Default',
                term           TEXT,
                arabic         TEXT,
                definition     TEXT,
                example        TEXT,
                synonym        TEXT,
                source_context TEXT,
                timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        try:
            await db.execute("ALTER TABLE cart ADD COLUMN deck_name TEXT NOT NULL DEFAULT 'Default'")
        except aiosqlite.OperationalError:
            pass

        # New unique index: per (user, deck, term) — allows same term in different decks
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_deck_term
            ON cart(user_id, deck_name, LOWER(term))
        """)

        # Attempt to drop the old global (user, term) index that is now too restrictive.
        # If it doesn't exist (new installs), the except swallows it silently.
        try:
            await db.execute("DROP INDEX idx_user_term")
        except aiosqlite.OperationalError:
            pass  # already gone or never existed

        # ── exported_words (vault) table ──────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS exported_words (
                user_id   INTEGER,
                deck_name TEXT NOT NULL DEFAULT 'Default',
                term      TEXT
            )
        """)

        try:
            await db.execute("ALTER TABLE exported_words ADD COLUMN deck_name TEXT NOT NULL DEFAULT 'Default'")
        except aiosqlite.OperationalError:
            pass

        # New unique index: per (user, deck, term)
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_exported_user_deck_term
            ON exported_words(user_id, deck_name, LOWER(term))
        """)

        try:
            await db.execute("DROP INDEX idx_exported_user_term")
        except aiosqlite.OperationalError:
            pass

        await db.commit()

    logger.info("Database initialized successfully.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  User CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def get_user(db_path: str, user_id: int) -> dict | None:
    """Fetch a user row by Telegram user_id.

    Returns a dict with all user columns or None if not found.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT user_id, main_deck, last_activity, total_exported,
                   gemini_key, selected_model, active_deck
            FROM users WHERE user_id = ?
            """,
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)


async def upsert_user(db_path: str, user_id: int, main_deck: str) -> None:
    """Insert a new user or update an existing user's main deck name.

    Also sets last_activity to the current UTC timestamp.
    Does NOT overwrite gemini_key, selected_model, or active_deck.
    """
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, main_deck, last_activity, total_exported)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(user_id) DO UPDATE SET
                main_deck     = excluded.main_deck,
                last_activity = excluded.last_activity
            """,
            (user_id, main_deck, now),
        )
        await db.commit()

    logger.info("User %d upserted with deck: %s", user_id, main_deck)


async def update_last_activity(db_path: str, user_id: int) -> None:
    """Touch the last_activity timestamp for a user."""
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET last_activity = ? WHERE user_id = ?",
            (now, user_id),
        )
        await db.commit()


async def increment_total_exported(db_path: str, user_id: int, count: int) -> None:
    """Increment the historical count of exported words for a user."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET total_exported = total_exported + ? WHERE user_id = ?",
            (count, user_id),
        )
        await db.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dynamic API Key & Model Settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def set_user_key(db_path: str, user_id: int, key: str | None) -> None:
    """Save (or clear) a user's personal Gemini API key.

    Pass key=None to delete the custom key and revert to server default.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET gemini_key = ? WHERE user_id = ?",
            (key, user_id),
        )
        await db.commit()

    if key:
        logger.info("User %d set a custom Gemini key.", user_id)
    else:
        logger.info("User %d cleared their Gemini key.", user_id)


async def set_user_model(db_path: str, user_id: int, model: str) -> None:
    """Save a user's preferred Gemini model string.

    Example values: 'gemini-2.5-flash', 'models/gemini-2.5-pro'.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET selected_model = ? WHERE user_id = ?",
            (model, user_id),
        )
        await db.commit()

    logger.info("User %d set model to: %s", user_id, model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Deck Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def set_active_deck(db_path: str, user_id: int, deck_name: str) -> None:
    """Switch the user's active deck.

    The deck is created implicitly the first time a word is added to it.
    This only persists the pointer in the users table.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET active_deck = ? WHERE user_id = ?",
            (deck_name, user_id),
        )
        await db.commit()

    logger.info("User %d switched active deck to: %s", user_id, deck_name)


async def get_active_deck(db_path: str, user_id: int) -> str:
    """Return the user's current active deck name.

    Falls back to 'Default' if the user row doesn't exist yet
    or the column is NULL.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT active_deck FROM users WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None or row[0] is None:
                return "Default"
            return row[0]


async def get_user_decks(db_path: str, user_id: int) -> list[dict]:
    """Return all decks a user has items in (cart only), with word counts.

    Returns a list of dicts: [{deck_name, count}, ...], ordered by deck_name.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """
            SELECT deck_name, COUNT(*) AS count
            FROM cart
            WHERE user_id = ?
            GROUP BY deck_name
            ORDER BY deck_name ASC
            """,
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"deck_name": row[0], "count": row[1]} for row in rows]


async def get_exported_stats(db_path: str, user_id: int) -> list[dict]:
    """Return exported word counts grouped by deck from the vault.

    Returns a list of dicts: [{deck_name, count}, ...], ordered by count desc.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """
            SELECT deck_name, COUNT(*) AS count
            FROM exported_words
            WHERE user_id = ?
            GROUP BY deck_name
            ORDER BY count DESC
            """,
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"deck_name": row[0], "count": row[1]} for row in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cart Operations (all scoped to active_deck)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def add_to_cart(
    db_path: str,
    user_id: int,
    deck_name: str,
    term: str,
    arabic: str,
    definition: str,
    example: str,
    synonym: str,
    source_context: str,
) -> bool:
    """Insert a word into the user's active-deck cart.

    Duplicate check is scoped to (user_id, deck_name, LOWER(term)).
    Also skips words already in the vault for this deck.

    Returns True if inserted, False if a duplicate.
    """
    term = term.strip().lower()

    async with aiosqlite.connect(db_path) as db:
        # Check if the word is already in the vault for this deck
        async with db.execute(
            """
            SELECT 1 FROM exported_words
            WHERE user_id = ? AND deck_name = ? AND LOWER(term) = ?
            """,
            (user_id, deck_name, term),
        ) as cursor:
            if await cursor.fetchone():
                logger.info(
                    "Cart dup (vault): user=%d deck=%s term=%s", user_id, deck_name, term
                )
                return False

        cursor = await db.execute(
            """
            INSERT INTO cart
                (user_id, deck_name, term, arabic, definition, example, synonym, source_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (user_id, deck_name, term, arabic, definition, example, synonym, source_context),
        )
        await db.commit()
        inserted = cursor.rowcount > 0

    if inserted:
        logger.info("Cart + : user=%d deck=%s term=%s", user_id, deck_name, term)
    else:
        logger.info("Cart dup: user=%d deck=%s term=%s (skipped)", user_id, deck_name, term)

    return inserted


async def get_cart(db_path: str, user_id: int, deck_name: str) -> list[dict]:
    """Return all cart items for a user in the given deck, ordered by insertion time."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, deck_name, term, arabic, definition, example,
                   synonym, source_context, timestamp
            FROM cart
            WHERE user_id = ? AND deck_name = ?
            ORDER BY id ASC
            """,
            (user_id, deck_name),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_cart_count(db_path: str, user_id: int, deck_name: str | None = None) -> int:
    """Return the number of items in a user's cart.

    If deck_name is provided, count only that deck.
    If deck_name is None, count across all decks.
    """
    async with aiosqlite.connect(db_path) as db:
        if deck_name is not None:
            async with db.execute(
                "SELECT COUNT(*) FROM cart WHERE user_id = ? AND deck_name = ?",
                (user_id, deck_name),
            ) as cursor:
                row = await cursor.fetchone()
        else:
            async with db.execute(
                "SELECT COUNT(*) FROM cart WHERE user_id = ?",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0]


async def clear_cart(db_path: str, user_id: int, deck_name: str) -> int:
    """Delete all cart items for a user in the given deck.

    Returns the number of rows deleted.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM cart WHERE user_id = ? AND deck_name = ?",
            (user_id, deck_name),
        )
        await db.commit()
        deleted = cursor.rowcount

    logger.info("Cart cleared: user=%d deck=%s deleted=%d", user_id, deck_name, deleted)
    return deleted


async def remove_from_cart(db_path: str, user_id: int, deck_name: str, term: str) -> bool:
    """Delete a specific word from the user's cart in the given deck.

    Returns True if deleted, False if not found.
    """
    term = term.strip().lower()

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM cart WHERE user_id = ? AND deck_name = ? AND LOWER(term) = ?",
            (user_id, deck_name, term),
        )
        await db.commit()
        deleted = cursor.rowcount > 0

    if deleted:
        logger.info("Cart - : user=%d deck=%s term=%s", user_id, deck_name, term)
    return deleted


async def archive_cart_to_vault(db_path: str, user_id: int, deck_name: str) -> None:
    """Move all terms from the active deck's cart into the permanent exported_words vault."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO exported_words (user_id, deck_name, term)
            SELECT user_id, deck_name, LOWER(term)
            FROM cart
            WHERE user_id = ? AND deck_name = ?
            ON CONFLICT DO NOTHING
            """,
            (user_id, deck_name),
        )
        await db.commit()

    logger.info("Archived cart to vault: user=%d deck=%s", user_id, deck_name)
