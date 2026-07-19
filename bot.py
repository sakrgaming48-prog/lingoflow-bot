"""
LingoFlow — Telegram Bot Core
===============================
Multi-Deck, Dynamic-API architecture.

Handles all Telegram interaction:
  • /start ConversationHandler (deck name onboarding)
  • Photo handler — Gemini Vision extraction pipeline
  • /cart — view active-deck cart contents
  • /clear — clear active-deck cart with inline keyboard confirmation
  • /remove — remove a specific word from the active-deck cart
  • /export — export active-deck cart as a named .apkg file
  • /deck — switch active deck
  • /decks — list all user decks with word counts
  • /setkey — save / clear personal Gemini API key
  • /setmodel — save preferred Gemini model string
  • /stats — per-deck breakdown of exported words
  • /help — usage guide
  • Lazy 24-hour reminder system
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta

import httpx
from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from anki_utils import generate_apkg

from db import (
    add_to_cart,
    archive_cart_to_vault,
    clear_cart,
    get_active_deck,
    get_cart,
    get_cart_count,
    get_exported_stats,
    get_user,
    get_user_decks,
    increment_total_exported,
    init_db,
    remove_from_cart,
    set_active_deck,
    set_user_key,
    set_user_model,
    update_last_activity,
    upsert_user,
)

# ── Environment ──────────────────────────────────────────────
load_dotenv()

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
DB_PATH: str = os.getenv("DB_PATH", "lingoflow.db")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("lingoflow")

# ── Conversation States ─────────────────────────────────────
AWAITING_DECK_NAME = 0

# ── Media Group Debounce Cache ──────────────────────────────
_MEDIA_GROUPS: dict[str, dict] = {}

# ── User Sessions (Conversational Memory) ───────────────────
USER_SESSIONS: dict[int, list[str]] = {}

# Fallback model names, ordered from most-preferred to least.
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

# Set by _init_gemini_model() during post_init — holds the server-default model name
DISCOVERED_MODEL_NAME: str | None = None


# ── Model Discovery Helpers ──────────────────────────────────


def _extract_version(model_name: str) -> float:
    """Extract the numeric version from a model name string.

    Examples:
      'models/gemini-2.5-flash'     → 2.5
      'models/gemini-3.1-flash-lite' → 3.1
      'gemini-1.5-flash-latest'     → 1.5
      'unknown-format'              → 0.0
    """
    match = re.search(r"(\d+(?:\.\d+)?)-flash", model_name)
    if match:
        return float(match.group(1))
    return 0.0


def _discover_flash_model() -> str | None:
    """Query the live model catalog and pick the best flash model via REST API.

    Selection criteria (in priority order):
      1. Supports 'generateContent'
      2. Name contains 'flash' (speed-optimized for our vision pipeline)
      3. Name does NOT contain 'preview', 'experimental', or 'image'
      4. Highest version number wins (e.g., 3.1 > 2.5 > 1.5)

    Returns the model name string, or None if discovery fails entirely.
    """
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        params = {"key": GEMINI_API_KEY}
        r = httpx.get(url, params=params, timeout=15.0)
        r.raise_for_status()
        data = r.json()

        candidates = []
        for m in data.get("models", []):
            supported_methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in supported_methods:
                continue
            name = m.get("name", "")
            name_lower = name.lower()
            if "flash" not in name_lower:
                continue
            if any(tag in name_lower for tag in ("preview", "experimental", "image")):
                continue
            version = _extract_version(name)
            candidates.append((version, name))
            logger.info("  Candidate: %-40s (version=%.1f)", name, version)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        winner = candidates[0][1]
        return winner

    except Exception:
        logger.exception("Model discovery failed")
        return None


def _init_gemini_model() -> str:
    """Discover and return the best available model name via REST checks.

    Returns a model name string (not a GenerativeModel object),
    which is stored in DISCOVERED_MODEL_NAME for use as the
    server-default fallback.
    """
    discovered = _discover_flash_model()
    if discovered:
        logger.info("✅ Model discovered: %s", discovered)
        return discovered

    logger.warning("Dynamic discovery found no suitable model. Trying fallbacks...")
    for name in _FALLBACK_MODELS:
        try:
            if name.startswith("models/"):
                url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}"
            params = {"key": GEMINI_API_KEY}
            r = httpx.get(url, params=params, timeout=10.0)
            r.raise_for_status()
            logger.info("✅ Fallback model OK: %s", name)
            return name
        except Exception:
            logger.warning("  Fallback %-35s → FAILED", name)
            continue

    logger.error("All fallbacks failed. Using %s (unverified).", _FALLBACK_MODELS[0])
    return _FALLBACK_MODELS[0]


async def _get_gemini_credentials_for_user(user_id: int) -> tuple[str, str]:
    """Return the (api_key, model_name) to use for this user.

    Priority:
      1. User has both gemini_key AND selected_model → use them.
      2. User has only gemini_key → use it with the server-discovered model.
      3. User has only selected_model → use server key with their model.
      4. Neither → use server key + discovered model.
    """
    user = await get_user(DB_PATH, user_id)

    user_key = user.get("gemini_key") if user else None
    user_model = user.get("selected_model") if user else None

    effective_key = user_key if user_key else GEMINI_API_KEY
    effective_model = user_model if user_model else (DISCOVERED_MODEL_NAME or _FALLBACK_MODELS[0])
    return effective_key, effective_model


# ── Gemini System Prompt ────────────────────────────────────
EXTRACTION_PROMPT = """You are a specialized vocabulary extraction engine for an Arabic-speaking medical student.

TASK:
Analyze the provided screenshot(s) and extract English vocabulary words/phrases from them.

RULES:
1. Extract ENGLISH vocabulary only. Completely ignore any Arabic text visible in the images.
2. Ignore UI elements (buttons, navigation, headers, app chrome).
3. The default context is General, Everyday, and Slang English (e.g., language learning apps like Taleek). ONLY apply medical context and medical definitions if the image is definitively from a medical textbook. Be exhaustive and extract up to 30 words per image.
4. Never use literal Arabic translation. Use pragmatic, contextual meanings suitable for everyday conversations.
5. For general context: lemmatize words to their base form (e.g., "running" → "run"). Exception: preserve phrasal verbs and idioms as-is (e.g., "give up", "break down").
6. For medical context: preserve medical collocations exactly as they appear (e.g., "running suture", "culture medium"). The word "culture" in medical context = مزرعة بكتيرية, NOT ثقافة.
7. If text is cut off at the edge, only extract complete, fully visible words. Ignore partial words.
8. Provide for each word:
   - "term": the English word or phrase (base form)
   - "arabic": accurate Arabic translation appropriate to the detected context
   - "definition": concise English definition (1 sentence)
   - "example": a natural example sentence using the word
   - "synonym": one English synonym (or "N/A" if none fits)
   - "source_context": "general" or "medical"

OUTPUT FORMAT:
Return ONLY a valid JSON array containing the combined vocabulary from all images. No markdown, no code fences, no conversational text, no explanation.
Example:
[
  {
    "term": "scrutinize",
    "arabic": "يفحص بدقة",
    "definition": "To examine something very carefully.",
    "example": "The doctor scrutinized the lab results before making a diagnosis.",
    "synonym": "examine",
    "source_context": "general"
  }
]

If you cannot extract any words from the images, return an empty array: []
"""

# Required keys that every extracted word must have
_REQUIRED_KEYS = {"term", "arabic", "definition", "example", "synonym", "source_context"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Gemini Vision — helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _parse_gemini_json(raw_text: str) -> list[dict]:
    """Parse Gemini's response into a list of word dicts.

    Handles common edge cases:
      - Response wrapped in ```json ... ``` code fences
      - Leading/trailing whitespace
      - Validates that each item has all required keys
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError("Gemini response is not a JSON array")

    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not _REQUIRED_KEYS.issubset(item.keys()):
            missing = _REQUIRED_KEYS - item.keys()
            logger.warning("Skipping item missing keys %s: %s", missing, item)
            continue
        item["term"] = str(item["term"]).strip().lower()
        item["arabic"] = str(item["arabic"]).strip()
        item["definition"] = str(item["definition"]).strip()
        item["example"] = str(item["example"]).strip()
        item["synonym"] = str(item["synonym"]).strip()
        item["source_context"] = str(item["source_context"]).strip().lower()
        if item["source_context"] not in ("general", "medical"):
            item["source_context"] = "general"
        if item["term"]:
            validated.append(item)

    return validated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /start — Onboarding ConversationHandler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def start_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Entry point for /start. Checks if the user already exists."""
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)

    if user and user["main_deck"]:
        await update.message.reply_text(
            f"مرحباً مجدداً! 👋\n"
            f"مجموعتك الرئيسية الحالية: <b>{user['main_deck']}</b>\n\n"
            f"أرسل اسماً جديداً لتغييرها، أو /cancel للإلغاء.",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            "مرحباً يا دكتور! 👋\n\n"
            "أنا <b>LingoFlow</b>، مساعدك الذكي لتحويل الكلمات "
            "من الصور إلى بطاقات Anki.\n\n"
            "أولاً، ما اسم مجموعة Anki الرئيسية؟\n"
            "(مثال: <code>English</code>)",
            parse_mode="HTML",
        )

    return AWAITING_DECK_NAME


async def receive_deck_name(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handles the user's deck name input. Validates and saves."""
    user_id = update.effective_user.id
    deck_name = update.message.text.strip()

    if not deck_name:
        await update.message.reply_text(
            "⚠️ الاسم لا يمكن أن يكون فارغاً. حاول مرة أخرى:"
        )
        return AWAITING_DECK_NAME

    if len(deck_name) > 50:
        await update.message.reply_text(
            "⚠️ الاسم طويل جداً (الحد الأقصى 50 حرفاً). حاول مرة أخرى:"
        )
        return AWAITING_DECK_NAME

    if "::" in deck_name:
        await update.message.reply_text(
            "⚠️ لا يمكن استخدام <code>::</code> في الاسم "
            "(محجوز لفواصل Anki). حاول مرة أخرى:",
            parse_mode="HTML",
        )
        return AWAITING_DECK_NAME

    await upsert_user(DB_PATH, user_id, deck_name)

    await update.message.reply_text(
        f"✅ تم حفظ المجموعة الرئيسية: <b>{deck_name}</b>\n\n"
        "مجموعتك النشطة الآن هي <b>Default</b>. "
        "يمكنك تغييرها في أي وقت عبر /deck\n\n"
        "أنت جاهز الآن! أرسل صورة وسأستخرج الكلمات منها. 📸",
        parse_mode="HTML",
    )

    return ConversationHandler.END


async def cancel_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handles /cancel inside the conversation."""
    await update.message.reply_text("تم الإلغاء. ✋")
    return ConversationHandler.END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lazy 24-Hour Reminder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def _check_reminder(user_id: int) -> str:
    """Check if the user's last activity was >24h ago with a non-empty cart.

    Returns the reminder string to prepend, or empty string if no reminder.
    Called before updating last_activity so the check uses the previous timestamp.
    """
    user = await get_user(DB_PATH, user_id)
    if not user or not user["last_activity"]:
        return ""

    try:
        last = datetime.fromisoformat(user["last_activity"])
        now = datetime.now(timezone.utc)
        if (now - last) > timedelta(hours=24):
            count = await get_cart_count(DB_PATH, user_id)
            if count > 0:
                return (
                    "💡 سلتك تحتوي على كلمات من الجلسة السابقة. "
                    "لا تنسَ تصديرها عبر /export\n\n"
                )
    except (ValueError, TypeError):
        logger.warning("Could not parse last_activity for user %d", user_id)

    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Photo Handler — Gemini Vision Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GeminiResponse:
    def __init__(self, text: str):
        self.text = text


async def _generate_content_with_fallback(
    prompt_parts: list, user_id: int
) -> GeminiResponse:
    """Attempt generation using the user's configured model, then server fallbacks.
    Uses direct REST API calls to bypass the google-generativeai SDK bug.
    """
    effective_key, user_model = await _get_gemini_credentials_for_user(user_id)

    # Build a deduplicated fallback queue starting from the user's model
    fallback_queue = [user_model] + [
        m for m in _FALLBACK_MODELS if m != user_model
    ]

    # Convert prompt parts to Google API JSON structure
    parts = []
    for part in prompt_parts:
        if isinstance(part, str):
            parts.append({"text": part})
        elif isinstance(part, dict) and "data" in part:
            mime_type = part.get("mime_type", "image/jpeg")
            raw_bytes = part["data"]
            b64_data = base64.b64encode(raw_bytes).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": b64_data
                }
            })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 4096
        }
    }

    last_exception = None

    for model_name in fallback_queue:
        if model_name.startswith("models/"):
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        headers = {
            "x-goog-api-key": effective_key,
            "Content-Type": "application/json"
        }

        logger.info("Attempting REST generation with model: %s", model_name)
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract response text
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return GeminiResponse(text)
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Model %s attempt %d failed: %s. Retrying in 2s...",
                    model_name, attempt, e
                )
                await asyncio.sleep(2.0)

    logger.error("All models and retries exhausted via REST. Last exception: %s", last_exception)
    raise last_exception


async def _send_reply_with_retry(message, text: str, max_retries: int = 3) -> None:
    """Safely send a message with a retry loop on Timeout/NetworkError."""
    for attempt in range(1, max_retries + 1):
        try:
            await message.reply_text(text, parse_mode="HTML")
            return
        except (TimedOut, NetworkError) as e:
            if attempt < max_retries:
                logger.warning("Telegram reply timed out. Retrying in 3s... (%s)", e)
                await asyncio.sleep(3.0)
            else:
                logger.error("Failed to send reply after %d attempts: %s", max_retries, e)


async def _process_images(
    user_id: int,
    file_ids: list[str],
    message,
    reminder: str,
    context: ContextTypes.DEFAULT_TYPE,
    caption: str = None,
) -> None:
    """Core extraction logic for one or more images."""
    try:
        USER_SESSIONS[user_id] = file_ids

        # ── Resolve active deck ───────────────────────────────
        deck_name = await get_active_deck(DB_PATH, user_id)

        # ── Download photos ───────────────────────────────────
        image_parts = []
        for file_id in file_ids:
            file = await context.bot.get_file(file_id)
            buf = io.BytesIO()
            await file.download_to_memory(buf)
            image_parts.append({
                "mime_type": "image/jpeg",
                "data": buf.getvalue(),
            })

        logger.info(
            "Processing %d photos for user=%d deck=%s",
            len(image_parts), user_id, deck_name
        )

        # ── Send to Gemini ───────────────────────────────────
        active_prompt = EXTRACTION_PROMPT
        if caption:
            active_prompt += (
                f"\n\nUSER CUSTOM INSTRUCTION FOR THIS BATCH: {caption}\n"
                f"Prioritize this instruction over general rules."
            )

        prompt = [active_prompt] + image_parts
        response = await _generate_content_with_fallback(prompt, user_id)

        raw_text = response.text
        logger.info("Gemini raw response length: %d chars", len(raw_text))

        # ── Parse JSON ───────────────────────────────────────
        words = _parse_gemini_json(raw_text)

        if not words:
            await _send_reply_with_retry(
                message,
                "🔍 لم أتمكن من العثور على كلمات إنجليزية في هذه الصورة/الصور.\n"
                "حاول إرسال صور تحتوي على نص إنجليزي واضح.",
            )
            return

        # ── Save to cart ─────────────────────────────────────
        inserted_words = []
        dup_count = 0
        seen_terms = set()
        for word in words:
            if word["term"] in seen_terms:
                dup_count += 1
                continue
            seen_terms.add(word["term"])

            inserted = await add_to_cart(
                DB_PATH,
                user_id,
                deck_name=deck_name,
                term=word["term"],
                arabic=word["arabic"],
                definition=word["definition"],
                example=word["example"],
                synonym=word["synonym"],
                source_context=word["source_context"],
            )
            if inserted:
                inserted_words.append(word)
            else:
                dup_count += 1

        # ── Determine context label ──────────────────────────
        contexts = {w["source_context"] for w in inserted_words} if inserted_words else set()
        if "medical" in contexts and "general" in contexts:
            context_label = "مختلط"
        elif "medical" in contexts:
            context_label = "طبي"
        else:
            context_label = "لغة عامة"

        # ── Build response ───────────────────────────────────
        term_list = "\n".join(
            f"  {i}. {w['term']}" for i, w in enumerate(inserted_words, 1)
        )

        parts = []
        if reminder:
            parts.append(reminder)

        parts.append(f"📂 المجموعة النشطة: <b>{deck_name}</b>")

        if inserted_words:
            parts.append(
                f"✅ تم إضافة <b>{len(inserted_words)}</b> كلمة جديدة "
                f"(سياق: {context_label}) لسلتك."
            )
        else:
            parts.append("ℹ️ لم يتم إضافة كلمات جديدة لسلتك.")

        if dup_count > 0:
            parts.append(
                f"ℹ️ {dup_count} كلمة تم حفظها مسبقاً (تم تخطيها)."
            )

        if term_list:
            parts.append(f"\n{term_list}")
        parts.append("\nاستمر في العمل الممتاز، دكتور. 💪")

        await _send_reply_with_retry(message, "\n".join(parts))

    except json.JSONDecodeError:
        logger.exception("Failed to parse Gemini JSON for user %d", user_id)
        await _send_reply_with_retry(
            message,
            "عذراً، لم أتمكن من قراءة الصورة بوضوح. "
            "يرجى إرسال صورة أوضح. 🔄",
        )

    except Exception:
        logger.exception("Gemini pipeline error for user %d", user_id)
        await _send_reply_with_retry(
            message,
            "عذراً، حدث خطأ أثناء معالجة الصور. "
            "يرجى المحاولة مرة أخرى. 🔄",
        )


async def _process_media_group_task(
    media_group_id: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Debounce task for media groups (albums). Waits 2 seconds then processes."""
    await asyncio.sleep(2.0)

    group_data = _MEDIA_GROUPS.pop(media_group_id, None)
    if not group_data:
        return

    file_ids = group_data["file_ids"]
    user_id = group_data["user_id"]
    message = group_data["message"]
    reminder = group_data["reminder"]
    caption = group_data.get("caption")

    await _process_images(user_id, file_ids, message, reminder, context, caption)


async def photo_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Process photos. Handles both single images and albums via debouncing."""
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)

    if not user or not user["main_deck"]:
        await update.message.reply_text(
            "⚠️ يجب إعداد مجموعة Anki أولاً.\n"
            "أرسل /start للبدء."
        )
        return

    reminder = await _check_reminder(user_id)
    await update_last_activity(DB_PATH, user_id)

    photo = update.message.photo[-1]
    file_id = photo.file_id
    media_group_id = update.message.media_group_id
    caption = update.message.caption

    if media_group_id:
        if media_group_id not in _MEDIA_GROUPS:
            _MEDIA_GROUPS[media_group_id] = {
                "file_ids": [file_id],
                "user_id": user_id,
                "message": update.message,
                "reminder": reminder,
                "caption": caption,
            }
            asyncio.create_task(
                _process_media_group_task(media_group_id, update, context)
            )
        else:
            _MEDIA_GROUPS[media_group_id]["file_ids"].append(file_id)
            if caption and not _MEDIA_GROUPS[media_group_id].get("caption"):
                _MEDIA_GROUPS[media_group_id]["caption"] = caption
        return

    await _process_images(user_id, [file_id], update.message, reminder, context, caption)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /help Command
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/help — Explain how to use the bot and all available commands."""
    text = (
        "📖 <b>دليل استخدام LingoFlow</b>\n\n"
        "1️⃣ أرسل صورة أو مجموعة صور (ألبوم) تحتوي على نص إنجليزي.\n"
        "2️⃣ سيقوم الذكاء الاصطناعي باستخراج الكلمات وترجمتها حسب السياق.\n"
        "3️⃣ الكلمات تُحفظ في السلة الخاصة بمجموعتك النشطة.\n"
        "4️⃣ استخدم /export لتصديرها كملف <code>.apkg</code>.\n\n"
        "🗂 <b>إدارة المجموعات:</b>\n"
        "  /deck [اسم] — تبديل المجموعة النشطة\n"
        "  /decks — عرض جميع مجموعاتك مع عدد الكلمات\n\n"
        "🛒 <b>إدارة السلة:</b>\n"
        "  /cart — عرض كلمات السلة الحالية\n"
        "  /remove [كلمة] — حذف كلمة معينة\n"
        "  /clear — مسح السلة بالكامل\n\n"
        "🤖 <b>إعدادات الذكاء الاصطناعي:</b>\n"
        "  /setkey [مفتاح] — حفظ مفتاح Gemini الشخصي (فارغ لإلغائه)\n"
        "  /setmodel [نموذج] — تحديد نموذج Gemini (مثال: gemini-2.5-flash)\n\n"
        "📊 /stats — تقرير الكلمات المُصدَّرة مقسَّم حسب المجموعة\n\n"
        "💡 <b>ميزة متقدمة:</b> يمكنك كتابة تعليق مع الصورة لتوجيه الذكاء الاصطناعي، "
        "أو إرسال رسالة نصية بعد الاستخراج لتعديل النتائج."
    )
    await update.message.reply_text(text, parse_mode="HTML")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /stats Command — per-deck breakdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def stats_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/stats — Show exported word breakdown by deck + totals."""
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)

    if not user:
        await update.message.reply_text("لم يتم العثور على بيانات. أرسل /start أولاً.")
        return

    total_exported = user.get("total_exported", 0)
    deck_stats = await get_exported_stats(DB_PATH, user_id)
    active_deck = await get_active_deck(DB_PATH, user_id)
    cart_count = await get_cart_count(DB_PATH, user_id, deck_name=active_deck)

    lines = [f"📊 <b>إحصائياتك:</b>\n"]

    if deck_stats:
        lines.append("📚 <b>الكلمات المُصدَّرة حسب المجموعة:</b>")
        for entry in deck_stats:
            lines.append(f"  • {entry['deck_name']}: <b>{entry['count']}</b> كلمة")
        lines.append(f"\n🌟 <b>المجموع الكلي:</b> {total_exported} كلمة")
    else:
        lines.append("لم تقم بتصدير أي كلمات بعد.")

    lines.append(f"\n🛒 في سلة «{active_deck}» الآن: <b>{cart_count}</b> كلمة")
    lines.append("\nاستمر في الإنجاز! 💪")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Text Handler (Conversational Memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def handle_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle follow-up instructions for the last uploaded image(s)."""
    user_id = update.effective_user.id
    text = update.message.text.strip()

    file_ids = USER_SESSIONS.get(user_id)
    if file_ids:
        reminder = await _check_reminder(user_id)
        await update_last_activity(DB_PATH, user_id)
        await _process_images(user_id, file_ids, update.message, reminder, context, text)
    else:
        await update.message.reply_text(
            "أرسل صورة أولاً لأتمكن من تحليلها وتطبيق تعليماتك عليها. 📸"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /cart — View cart contents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def cart_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/cart — Display a numbered list of terms in the active-deck cart."""
    user_id = update.effective_user.id

    reminder = await _check_reminder(user_id)
    await update_last_activity(DB_PATH, user_id)

    deck_name = await get_active_deck(DB_PATH, user_id)
    items = await get_cart(DB_PATH, user_id, deck_name)

    if not items:
        await update.message.reply_text(
            f"سلة «<b>{deck_name}</b>» فارغة. أرسل صورة أولاً. 📸",
            parse_mode="HTML",
        )
        return

    term_list = "\n".join(
        f"  {i}. {item['term']}" for i, item in enumerate(items, 1)
    )

    text = (
        f"{reminder}"
        f"📂 المجموعة النشطة: <b>{deck_name}</b>\n"
        f"🛒 السلة تحتوي على <b>{len(items)}</b> كلمة:\n\n"
        f"{term_list}\n\n"
        f"📤 أرسل /export لتصدير البطاقات\n"
        f"🗑 أرسل /clear لمسح السلة\n"
        f"➖ أرسل /remove [word] لحذف كلمة معينة"
    )

    await update.message.reply_text(text, parse_mode="HTML")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /remove — Delete a specific word
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def remove_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/remove — Delete a specific word from the active-deck cart."""
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text(
            "⚠️ يرجى تحديد الكلمة المراد حذفها.\n"
            "مثال: <code>/remove scrutinize</code>",
            parse_mode="HTML",
        )
        return

    term = " ".join(context.args).strip()
    deck_name = await get_active_deck(DB_PATH, user_id)
    deleted = await remove_from_cart(DB_PATH, user_id, deck_name, term)

    if deleted:
        await update.message.reply_text(
            f"🗑️ تم حذف الكلمة: <b>{term}</b> من سلة «{deck_name}».",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"لم يتم العثور على الكلمة: <b>{term}</b> في سلة «{deck_name}».",
            parse_mode="HTML",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /clear — Clear cart with inline keyboard confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CB_CLEAR_CONFIRM = "clear_confirm"
_CB_CLEAR_CANCEL = "clear_cancel"


async def clear_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/clear — Present inline keyboard to confirm active-deck cart deletion."""
    user_id = update.effective_user.id
    deck_name = await get_active_deck(DB_PATH, user_id)
    count = await get_cart_count(DB_PATH, user_id, deck_name)

    if count == 0:
        await update.message.reply_text(
            f"سلة «<b>{deck_name}</b>» فارغة بالفعل. ✨",
            parse_mode="HTML",
        )
        return

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ نعم، امسح", callback_data=_CB_CLEAR_CONFIRM),
            InlineKeyboardButton("❌ لا، عد للخلف", callback_data=_CB_CLEAR_CANCEL),
        ]
    ])

    await update.message.reply_text(
        f"هل أنت متأكد من مسح <b>{count}</b> كلمة من سلة «{deck_name}»؟",
        reply_markup=keyboard,
        parse_mode="HTML",
    )


async def clear_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle the inline keyboard response for /clear confirmation."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id

    if query.data == _CB_CLEAR_CONFIRM:
        deck_name = await get_active_deck(DB_PATH, user_id)
        deleted = await clear_cart(DB_PATH, user_id, deck_name)
        await query.edit_message_text(
            f"🗑 تم مسح <b>{deleted}</b> كلمة من سلة «{deck_name}».",
            parse_mode="HTML",
        )
    elif query.data == _CB_CLEAR_CANCEL:
        await query.edit_message_text("تم الإلغاء. سلتك لم تتأثر. ✋")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /deck — Switch active deck
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Allowed characters: letters, digits, spaces, underscores, hyphens
_DECK_NAME_RE = re.compile(r"^[\w\s\-]{1,30}$", re.UNICODE)


def _normalize_deck_name(raw: str) -> str | None:
    """Normalize and validate a deck name provided by the user.

    Rules:
      - Strip leading/trailing whitespace
      - Max 30 characters
      - Only alphanumeric, spaces, underscores, hyphens (Unicode-safe)
    Returns the normalized name, or None if invalid.
    """
    name = raw.strip()
    if not name:
        return None
    # Collapse internal whitespace
    name = re.sub(r"\s+", " ", name)
    if len(name) > 30:
        name = name[:30]
    if not _DECK_NAME_RE.match(name):
        return None
    return name


async def deck_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/deck [name] — Switch the active deck."""
    user_id = update.effective_user.id

    if not context.args:
        current = await get_active_deck(DB_PATH, user_id)
        decks = await get_user_decks(DB_PATH, user_id)
        deck_list = "\n".join(
            f"  • {d['deck_name']} ({d['count']} كلمة)" for d in decks
        ) or "  (لا توجد مجموعات بعد)"
        await update.message.reply_text(
            f"📂 مجموعتك النشطة الحالية: <b>{current}</b>\n\n"
            f"مجموعاتك:\n{deck_list}\n\n"
            f"لتبديل المجموعة: <code>/deck [اسم المجموعة]</code>",
            parse_mode="HTML",
        )
        return

    raw_name = " ".join(context.args)
    deck_name = _normalize_deck_name(raw_name)

    if deck_name is None:
        await update.message.reply_text(
            "⚠️ اسم المجموعة غير صالح.\n"
            "يجب أن يحتوي على حروف وأرقام ومسافات وشرطات سفلية فقط، "
            "بحد أقصى 30 حرفاً.\n"
            f"مثال: <code>/deck Pathology</code>",
            parse_mode="HTML",
        )
        return

    user = await get_user(DB_PATH, user_id)
    if not user:
        await update.message.reply_text("أرسل /start أولاً لإعداد حسابك.")
        return

    await set_active_deck(DB_PATH, user_id, deck_name)

    # Show how many words are already in this deck (may be 0 for new deck)
    count = await get_cart_count(DB_PATH, user_id, deck_name)
    count_note = (
        f"تحتوي هذه المجموعة حالياً على <b>{count}</b> كلمة في السلة."
        if count > 0 else
        "هذه مجموعة جديدة — ستُضاف كلماتك إليها من الآن."
    )

    await update.message.reply_text(
        f"✅ تم التبديل إلى المجموعة: <b>{deck_name}</b>\n"
        f"{count_note}",
        parse_mode="HTML",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /decks — List all user decks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def decks_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/decks — List all decks with their active cart word counts."""
    user_id = update.effective_user.id
    active_deck = await get_active_deck(DB_PATH, user_id)
    decks = await get_user_decks(DB_PATH, user_id)

    if not decks:
        await update.message.reply_text(
            "لا توجد مجموعات بعد. أرسل صورة أولاً لإضافة كلمات إلى سلتك. 📸"
        )
        return

    lines = ["📚 <b>مجموعاتك:</b>\n"]
    for d in decks:
        marker = " ✅" if d["deck_name"] == active_deck else ""
        lines.append(f"  • <b>{d['deck_name']}</b>: {d['count']} كلمة{marker}")

    lines.append(f"\n✅ = المجموعة النشطة الحالية")
    lines.append(f"لتبديل المجموعة: <code>/deck [اسم]</code>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /setkey — Save or clear personal Gemini API key
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def setkey_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/setkey [key] — Save a personal Gemini key. No args → clear it."""
    user_id = update.effective_user.id

    user = await get_user(DB_PATH, user_id)
    if not user:
        await update.message.reply_text("أرسل /start أولاً لإعداد حسابك.")
        return

    if not context.args:
        # Clear the key
        await set_user_key(DB_PATH, user_id, None)
        await update.message.reply_text(
            "🔑 تم حذف مفتاحك الشخصي. سيتم استخدام مفتاح الخادم الافتراضي من الآن.",
        )
        return

    key = context.args[0].strip()

    if not (key.startswith("AI") or key.startswith("AQ.")) or len(key) < 20:
        await update.message.reply_text(
            "⚠️ مفتاح Gemini غير صالح. يجب أن يبدأ بـ <code>AI</code> أو <code>AQ.</code> ويكون بطول مناسب.\n"
            "احصل على مفتاحك من: aistudio.google.com",
            parse_mode="HTML",
        )
        return

    await set_user_key(DB_PATH, user_id, key)

    # Delete the command message immediately to protect the key from chat history
    try:
        await update.message.delete()
    except Exception:
        pass

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🔑 تم حفظ مفتاح Gemini الشخصي بنجاح.\n"
             "⚠️ تم حذف رسالتك لحماية المفتاح من سجل المحادثة.",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /setmodel — Save preferred Gemini model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def setmodel_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/setmodel [model] — Set the Gemini model to use for generation."""
    user_id = update.effective_user.id

    user = await get_user(DB_PATH, user_id)
    if not user:
        await update.message.reply_text("أرسل /start أولاً لإعداد حسابك.")
        return

    if not context.args:
        current_model = user.get("selected_model") or DISCOVERED_MODEL_NAME or _FALLBACK_MODELS[0]
        await update.message.reply_text(
            f"🤖 نموذجك الحالي: <code>{current_model}</code>\n\n"
            f"لتغيير النموذج:\n<code>/setmodel gemini-2.5-flash</code>\n"
            f"أو:\n<code>/setmodel gemini-2.5-pro</code>",
            parse_mode="HTML",
        )
        return

    model_name = context.args[0].strip()

    # Normalize: if user omits the "models/" prefix, that's fine — genai handles it
    if not model_name:
        await update.message.reply_text("⚠️ يرجى تحديد اسم النموذج.")
        return

    await set_user_model(DB_PATH, user_id, model_name)

    await update.message.reply_text(
        f"🤖 تم حفظ النموذج: <code>{model_name}</code>\n"
        "سيُستخدم هذا النموذج في جميع طلباتك القادمة.",
        parse_mode="HTML",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /export — Generate Anki package for active deck
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def export_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/export — Export the active deck's cart as a named .apkg file.

    Deck hierarchy: MainDeck::ActiveDeck
    File name: LingoFlow_[ActiveDeck]_[Date].apkg
    """
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)

    if not user or not user["main_deck"]:
        await update.message.reply_text(
            "⚠️ يجب إعداد مجموعة Anki أولاً.\nأرسل /start للبدء."
        )
        return

    deck_name = await get_active_deck(DB_PATH, user_id)
    items = await get_cart(DB_PATH, user_id, deck_name)

    if not items:
        await update.message.reply_text(
            f"سلة «<b>{deck_name}</b>» فارغة. أرسل صورة أولاً. 📸",
            parse_mode="HTML",
        )
        return

    main_deck = user["main_deck"]
    date_str = datetime.now().strftime("%d%b")  # e.g. "19Jul"

    # Safe filename: replace spaces with underscores, strip special chars
    safe_deck = re.sub(r"[^\w\-]", "_", deck_name)
    filename = f"LingoFlow_{safe_deck}_{date_str}.apkg"
    output_path = os.path.join("exports", filename)

    os.makedirs("exports", exist_ok=True)
    status_msg = await update.message.reply_text("⏳ جاري تجهيز الملف...")

    try:
        generate_apkg(user_id, main_deck, deck_name, items, output_path)

        with open(output_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=filename,
                caption=(
                    f"✅ تم تصدير <b>{len(items)}</b> بطاقة من مجموعة «{deck_name}»!\n"
                    f"المجموعة في Anki: <code>{main_deck}::{deck_name}</code>\n"
                    "بالتوفيق في مذاكرتك يا دكتور! 🎓"
                ),
            )

        await increment_total_exported(DB_PATH, user_id, len(items))
        await archive_cart_to_vault(DB_PATH, user_id, deck_name)
        await clear_cart(DB_PATH, user_id, deck_name)
        await status_msg.delete()

    except Exception:
        logger.exception("Export failed for user=%d deck=%s", user_id, deck_name)
        await status_msg.edit_text("❌ حدث خطأ أثناء التصدير. يرجى المحاولة مرة أخرى.")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Application bootstrap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def post_init(application) -> None:
    """Runs after the Application is built but before polling starts.

    Initializes the database schema and discovers the best Gemini model.
    """
    global DISCOVERED_MODEL_NAME

    logger.info("Running post_init: initializing database...")
    await init_db(DB_PATH)

    logger.info("Running post_init: discovering Gemini model...")
    DISCOVERED_MODEL_NAME = _init_gemini_model()
    logger.info("Server-default model: %s", DISCOVERED_MODEL_NAME)

    logger.info("Running post_init: setting bot commands...")
    commands = [
        BotCommand("start",    "إعداد مجموعة Anki الرئيسية"),
        BotCommand("deck",     "تبديل المجموعة النشطة"),
        BotCommand("decks",    "عرض جميع مجموعاتك"),
        BotCommand("cart",     "عرض محتويات السلة الحالية"),
        BotCommand("remove",   "حذف كلمة معينة من السلة"),
        BotCommand("clear",    "مسح جميع الكلمات من السلة"),
        BotCommand("export",   "تصدير السلة إلى ملف Anki"),
        BotCommand("stats",    "تقرير الكلمات المُصدَّرة حسب المجموعة"),
        BotCommand("setkey",   "حفظ مفتاح Gemini الشخصي"),
        BotCommand("setmodel", "تحديد نموذج Gemini المفضل"),
        BotCommand("help",     "دليل الاستخدام والتعليمات"),
    ]
    await application.bot.set_my_commands(commands)

    logger.info("post_init complete.")


def main() -> None:
    """Build the Telegram Application, register handlers, and start polling."""

    if not BOT_TOKEN:
        logger.critical("BOT_TOKEN is not set. Check your .env file.")
        raise SystemExit(1)

    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY is not set. Check your .env file.")
        raise SystemExit(1)

    # ── Build application ────────────────────────────────────
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .connect_timeout(60.0)
        .read_timeout(60.0)
        .get_updates_connection_pool_size(10)
        .post_init(post_init)
        .build()
    )

    # ── /start ConversationHandler ───────────────────────────
    start_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            AWAITING_DECK_NAME: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, receive_deck_name
                )
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )

    # ── Register handlers (order matters!) ───────────────────
    application.add_handler(start_conv_handler)

    # Core cart / deck commands
    application.add_handler(CommandHandler("cart",     cart_command))
    application.add_handler(CommandHandler("remove",   remove_command))
    application.add_handler(CommandHandler("clear",    clear_command))
    application.add_handler(CommandHandler("export",   export_command))
    application.add_handler(CommandHandler("deck",     deck_command))
    application.add_handler(CommandHandler("decks",    decks_command))

    # AI configuration commands
    application.add_handler(CommandHandler("setkey",   setkey_command))
    application.add_handler(CommandHandler("setmodel", setmodel_command))

    # Info commands
    application.add_handler(CommandHandler("stats",    stats_command))
    application.add_handler(CommandHandler("help",     help_command))

    # Media and callback handlers
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(CallbackQueryHandler(
        clear_callback,
        pattern=f"^({_CB_CLEAR_CONFIRM}|{_CB_CLEAR_CANCEL})$",
    ))

    # Text Handler MUST be last — catches everything not handled above
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # ── Start polling ────────────────────────────────────────
    logger.info("LingoFlow bot starting... 🚀")

    while True:
        try:
            application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=())
            break
        except (TimedOut, NetworkError):
            logger.warning("⚠️ Telegram servers unreachable, retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            logger.error("Critical error during polling: %s", e)
            break
