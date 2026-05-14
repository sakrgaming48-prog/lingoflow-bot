"""
LingoFlow — Async Database Layer
=================================
Provides all database operations using aiosqlite.
Each function opens its own connection to avoid SQLite locking
issues in an async context.
"""

import aiosqlite
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def init_db(db_path: str) -> None:
    """Initialize the database schema.

    Creates the `users` and `cart` tables if they don't exist,
    along with the idx_user_term index for duplicate lookups.
    Called once on bot startup via the post_init hook.
    """
    logger.info("Initializing database at: %s", db_path)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id  INTEGER PRIMARY KEY,
                main_deck TEXT,
                last_activity TIMESTAMP,
                total_exported INTEGER DEFAULT 0
            )
        """)
        
        # Backward compatibility for existing databases
        try:
            await db.execute("ALTER TABLE users ADD COLUMN total_exported INTEGER DEFAULT 0")
        except aiosqlite.OperationalError:
            pass

        await db.execute("""
            CREATE TABLE IF NOT EXISTS cart (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER,
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

        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_term
            ON cart(user_id, term)
        """)

        await db.commit()

    logger.info("Database initialized successfully.")


async def get_user(db_path: str, user_id: int) -> dict | None:
    """Fetch a user row by Telegram user_id.

    Returns a dict with keys {user_id, main_deck, last_activity}
    or None if the user doesn't exist yet.
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, main_deck, last_activity, total_exported FROM users WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)


async def upsert_user(db_path: str, user_id: int, main_deck: str) -> None:
    """Insert a new user or update an existing user's main deck name.

    Also sets last_activity to the current UTC timestamp.
    Uses INSERT ... ON CONFLICT to handle both cases atomically.
    """
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, main_deck, last_activity, total_exported)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(user_id) DO UPDATE SET
                main_deck = excluded.main_deck,
                last_activity = excluded.last_activity
            """,
            (user_id, main_deck, now),
        )
        await db.commit()

    logger.info("User %d upserted with deck: %s", user_id, main_deck)


async def update_last_activity(db_path: str, user_id: int) -> None:
    """Touch the last_activity timestamp for a user.

    Called on any user interaction to support the lazy 24-hour
    reminder logic in Phase 3.
    """
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET last_activity = ? WHERE user_id = ?",
            (now, user_id),
        )
        await db.commit()


async def add_to_cart(
    db_path: str,
    user_id: int,
    term: str,
    arabic: str,
    definition: str,
    example: str,
    synonym: str,
    source_context: str,
) -> bool:
    """Insert a word into the user's cart.

    Uses INSERT OR IGNORE so that duplicate user+term pairs
    (enforced by the UNIQUE idx_user_term index) are silently
    skipped rather than raising an error.

    Returns True if the row was actually inserted, False if it
    was a duplicate that got skipped.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT OR IGNORE INTO cart
                (user_id, term, arabic, definition, example, synonym, source_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, term, arabic, definition, example, synonym, source_context),
        )
        await db.commit()
        inserted = cursor.rowcount > 0

    if inserted:
        logger.info("Cart + : user=%d term=%s", user_id, term)
    else:
        logger.info("Cart dup: user=%d term=%s (skipped)", user_id, term)

    return inserted


async def get_cart(db_path: str, user_id: int) -> list[dict]:
    """Return all cart items for a user, ordered by insertion time.

    Each item is a dict with keys:
      {id, term, arabic, definition, example, synonym, source_context, timestamp}
    """
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, term, arabic, definition, example,
                   synonym, source_context, timestamp
            FROM cart
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_cart_count(db_path: str, user_id: int) -> int:
    """Return the number of items in a user's cart.

    Uses COUNT(*) for efficiency — no need to fetch full rows.
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM cart WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0]


async def clear_cart(db_path: str, user_id: int) -> int:
    """Delete all cart items for a user.

    Returns the number of rows deleted.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM cart WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
        deleted = cursor.rowcount

    logger.info("Cart cleared: user=%d deleted=%d", user_id, deleted)
    return deleted


async def remove_from_cart(db_path: str, user_id: int, term: str) -> bool:
    """Delete a specific word from the user's cart.
    
    Returns True if deleted, False if not found.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM cart WHERE user_id = ? AND term = ?",
            (user_id, term),
        )
        await db.commit()
        deleted = cursor.rowcount > 0
        
    if deleted:
        logger.info("Cart - : user=%d term=%s", user_id, term)
    return deleted


async def increment_total_exported(db_path: str, user_id: int, count: int) -> None:
    """Increment the historical count of exported words for a user."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET total_exported = total_exported + ? WHERE user_id = ?",
            (count, user_id),
        )
        await db.commit()
