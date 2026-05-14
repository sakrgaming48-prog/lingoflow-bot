"""Phase 3 smoke tests — cart queries, clear, and 24h reminder logic."""

import asyncio
import os
from datetime import datetime, timezone, timedelta

from db import (
    init_db,
    upsert_user,
    add_to_cart,
    get_cart,
    get_cart_count,
    clear_cart,
    get_user,
    update_last_activity,
)
from bot import _check_reminder

DB = "_test_phase3.db"


async def setup():
    """Fresh DB + user + 3 cart items."""
    if os.path.exists(DB):
        os.remove(DB)
    # Monkey-patch DB_PATH for _check_reminder
    import bot
    bot.DB_PATH = DB

    await init_db(DB)
    await upsert_user(DB, 42, "English")
    await add_to_cart(DB, 42, "hello", "مرحبا", "greeting", "ex", "hi", "general")
    await add_to_cart(DB, 42, "world", "عالم", "the earth", "ex", "globe", "general")
    await add_to_cart(DB, 42, "test", "اختبار", "a trial", "ex", "exam", "general")


async def test_get_cart():
    items = await get_cart(DB, 42)
    assert len(items) == 3, f"Expected 3, got {len(items)}"
    assert items[0]["term"] == "hello"
    assert items[2]["term"] == "test"
    # Verify ordered by id ASC
    assert items[0]["id"] < items[1]["id"] < items[2]["id"]
    print("Test 1 (get_cart):              PASS ✅")


async def test_get_cart_count():
    count = await get_cart_count(DB, 42)
    assert count == 3, f"Expected 3, got {count}"
    # Non-existent user
    count2 = await get_cart_count(DB, 9999)
    assert count2 == 0
    print("Test 2 (get_cart_count):         PASS ✅")


async def test_get_cart_empty():
    items = await get_cart(DB, 9999)
    assert items == []
    print("Test 3 (get_cart empty user):    PASS ✅")


async def test_clear_cart():
    # Add items for a different user so we don't affect user 42
    await add_to_cart(DB, 100, "alpha", "أ", "d", "e", "s", "general")
    await add_to_cart(DB, 100, "beta", "ب", "d", "e", "s", "general")
    deleted = await clear_cart(DB, 100)
    assert deleted == 2, f"Expected 2 deleted, got {deleted}"
    remaining = await get_cart_count(DB, 100)
    assert remaining == 0
    # User 42's cart should be untouched
    assert await get_cart_count(DB, 42) == 3
    print("Test 4 (clear_cart):            PASS ✅")


async def test_clear_cart_already_empty():
    deleted = await clear_cart(DB, 9999)
    assert deleted == 0
    print("Test 5 (clear already empty):   PASS ✅")


async def test_reminder_no_trigger():
    """Activity was recent → no reminder."""
    await update_last_activity(DB, 42)  # sets to now
    reminder = await _check_reminder(42)
    assert reminder == "", f"Expected empty, got: {reminder!r}"
    print("Test 6 (reminder - recent):     PASS ✅")


async def test_reminder_triggers():
    """Activity was >24h ago with non-empty cart → reminder fires."""
    import aiosqlite
    old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    async with aiosqlite.connect(DB) as db:
        await db.execute(
            "UPDATE users SET last_activity = ? WHERE user_id = ?",
            (old_time, 42),
        )
        await db.commit()

    reminder = await _check_reminder(42)
    assert "💡" in reminder, f"Expected reminder, got: {reminder!r}"
    assert "/export" in reminder
    print("Test 7 (reminder - 25h ago):    PASS ✅")


async def test_reminder_empty_cart():
    """Activity was >24h ago but cart is empty → no reminder."""
    import aiosqlite
    await upsert_user(DB, 200, "Test")
    old_time = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    async with aiosqlite.connect(DB) as db:
        await db.execute(
            "UPDATE users SET last_activity = ? WHERE user_id = ?",
            (old_time, 200),
        )
        await db.commit()

    reminder = await _check_reminder(200)
    assert reminder == "", f"Expected empty (no cart items), got: {reminder!r}"
    print("Test 8 (reminder - empty cart):  PASS ✅")


async def main():
    await setup()
    await test_get_cart()
    await test_get_cart_count()
    await test_get_cart_empty()
    await test_clear_cart()
    await test_clear_cart_already_empty()
    await test_reminder_no_trigger()
    await test_reminder_triggers()
    await test_reminder_empty_cart()

    # Cleanup
    os.remove(DB)
    print("\n🎉 All Phase 3 tests passed!")


asyncio.run(main())
