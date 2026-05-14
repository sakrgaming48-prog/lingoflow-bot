"""
LingoFlow — Telegram Bot Core
===============================
Handles all Telegram interaction:
  • /start ConversationHandler (deck name onboarding)
  • Photo handler — Gemini Vision extraction pipeline
  • /cart — view cart contents
  • /clear — clear cart with inline keyboard confirmation
  • /export — placeholder (Phase 4)
  • Lazy 24-hour reminder system
"""

import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta

import google.generativeai as genai
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
    clear_cart,
    get_cart,
    get_cart_count,
    get_user,
    init_db,
    update_last_activity,
    upsert_user,
    increment_total_exported,
    remove_from_cart,
    archive_cart_to_vault,
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
AWAITING_SUBDECK_NAME = 1

# ── Media Group Debounce Cache ──────────────────────────────
_MEDIA_GROUPS: dict[str, dict] = {}

# ── User Sessions (Conversational Memory) ───────────────────
USER_SESSIONS: dict[int, list[str]] = {}

# ── Gemini Configuration ────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# Generation settings shared across all model instances
_GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.2,
    max_output_tokens=4096,
)

# Fallback model names, ordered from most-preferred to least.
# Updated May 2026 — older 1.5 names are 404-prone.
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

# Will be set by _init_gemini_model() during post_init
GEMINI_MODEL = None


def _discover_flash_model() -> str | None:
    """Query the live model catalog and pick the best flash model.

    Selection criteria (in priority order):
      1. Supports 'generateContent'
      2. Name contains 'flash' (speed-optimized for our vision pipeline)
      3. Name does NOT contain 'preview', 'experimental', or 'image'
         (we want a stable, text-output model)
      4. Highest version number wins (e.g., 3.1 > 2.5 > 1.5)

    Returns the model name string, or None if discovery fails entirely.
    """
    try:
        candidates = []
        for m in genai.list_models():
            if "generateContent" not in m.supported_generation_methods:
                continue
            name_lower = m.name.lower()
            # Must be a flash variant
            if "flash" not in name_lower:
                continue
            # Skip preview/experimental/image-generation models
            if any(tag in name_lower for tag in ("preview", "experimental", "image")):
                continue
            # Extract version number for sorting (e.g., "gemini-2.5-flash" → 2.5)
            version = _extract_version(m.name)
            candidates.append((version, m.name))
            logger.info("  Candidate: %-40s (version=%.1f)", m.name, version)

        if not candidates:
            return None

        # Highest version first
        candidates.sort(key=lambda c: c[0], reverse=True)
        winner = candidates[0][1]
        return winner

    except Exception:
        logger.exception("Model discovery failed")
        return None


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


def _init_gemini_model() -> genai.GenerativeModel:
    """Initialize the Gemini model with dynamic discovery + fallback chain.

    Called once during post_init. Logs the selected model name.
    """
    # Step 1: Try dynamic discovery
    discovered = _discover_flash_model()
    if discovered:
        logger.info("✅ Model discovered: %s", discovered)
        return genai.GenerativeModel(
            model_name=discovered,
            generation_config=_GENERATION_CONFIG,
        )

    # Step 2: Walk the fallback list
    logger.warning("Dynamic discovery found no suitable model. Trying fallbacks...")
    for name in _FALLBACK_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=name,
                generation_config=_GENERATION_CONFIG,
            )
            # Quick probe: calling count_tokens is cheap and confirms the model exists
            model.count_tokens("test")
            logger.info("✅ Fallback model OK: %s", name)
            return model
        except Exception:
            logger.warning("  Fallback %-35s → FAILED", name)
            continue

    # Step 3: Last resort — use the top fallback without probing
    logger.error("All fallbacks failed. Using %s (unverified).", _FALLBACK_MODELS[0])
    return genai.GenerativeModel(
        model_name=_FALLBACK_MODELS[0],
        generation_config=_GENERATION_CONFIG,
    )

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

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError("Gemini response is not a JSON array")

    # Validate and sanitize each entry
    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not _REQUIRED_KEYS.issubset(item.keys()):
            missing = _REQUIRED_KEYS - item.keys()
            logger.warning("Skipping item missing keys %s: %s", missing, item)
            continue
        # Normalize term to stripped lowercase for consistency
        item["term"] = str(item["term"]).strip().lower()
        item["arabic"] = str(item["arabic"]).strip()
        item["definition"] = str(item["definition"]).strip()
        item["example"] = str(item["example"]).strip()
        item["synonym"] = str(item["synonym"]).strip()
        item["source_context"] = str(item["source_context"]).strip().lower()
        if item["source_context"] not in ("general", "medical"):
            item["source_context"] = "general"
        if item["term"]:  # skip empty terms
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
        # Returning user — show current deck, offer to change
        await update.message.reply_text(
            f"مرحباً مجدداً! 👋\n"
            f"مجموعتك الحالية: <b>{user['main_deck']}</b>\n\n"
            f"أرسل اسماً جديداً لتغييرها، أو /cancel للإلغاء.",
            parse_mode="HTML",
        )
    else:
        # New user — welcome and prompt for deck name
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

    # ── Validation ───────────────────────────────────────────
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

    # ── Save to database ─────────────────────────────────────
    await upsert_user(DB_PATH, user_id, deck_name)

    await update.message.reply_text(
        f"✅ تم حفظ المجموعة: <b>{deck_name}</b>\n\n"
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
    Called before updating last_activity so the check uses the *previous* timestamp.
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

async def _generate_content_with_fallback(prompt_parts: list) -> any:
    """Wrapper to handle Google API rate limits and server errors with retries and model fallbacks."""
    fallback_queue = [
        GEMINI_MODEL.model_name,
        "models/gemini-3.1-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash"
    ]
    
    last_exception = None
    
    for model_name in fallback_queue:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=_GENERATION_CONFIG,
            )
        except Exception as e:
            logger.warning("Failed to initialize model %s: %s", model_name, e)
            continue
            
        logger.info("Attempting generation with model: %s", model_name)
        for attempt in range(1, 4):
            try:
                response = await model.generate_content_async(prompt_parts)
                return response
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Model %s attempt %d failed: %s. Retrying in 2 seconds...", 
                    model_name, attempt, e
                )
                await asyncio.sleep(2.0)
                
    logger.error("All models and retries failed. Last exception: %s", last_exception)
    raise last_exception


async def _send_reply_with_retry(message, text: str, max_retries: int = 3) -> None:
    """Wrapper to safely send a message with a retry loop on Timeout/NetworkError."""
    for attempt in range(1, max_retries + 1):
        try:
            await message.reply_text(text, parse_mode="HTML")
            return
        except (TimedOut, NetworkError) as e:
            if attempt < max_retries:
                logger.warning(f"Telegram reply timed out. Retrying in 3 seconds... ({e})")
                await asyncio.sleep(3.0)
            else:
                logger.error(f"Failed to send reply after {max_retries} attempts: {e}")

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
        # Save to conversational memory for follow-ups
        USER_SESSIONS[user_id] = file_ids

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
            "Processing %d photos for user=%d", len(image_parts), user_id
        )

        # ── Send to Gemini ───────────────────────────────────
        active_prompt = EXTRACTION_PROMPT
        if caption:
            active_prompt += (
                f"\n\nUSER CUSTOM INSTRUCTION FOR THIS BATCH: {caption}\n"
                f"Prioritize this instruction over general rules."
            )

        prompt = [active_prompt] + image_parts
        response = await _generate_content_with_fallback(prompt)

        raw_text = response.text
        logger.info("Gemini raw response length: %d chars", len(raw_text))

        # ── Parse JSON ───────────────────────────────────────
        words = _parse_gemini_json(raw_text)

        if not words:
            await _send_reply_with_retry(message,
                "🔍 لم أتمكن من العثور على كلمات إنجليزية في هذه الصورة/الصور.\n"
                "حاول إرسال صور تحتوي على نص إنجليزي واضح."
            )
            return

        # ── Save to cart ─────────────────────────────────────
        inserted_words = []
        dup_count = 0
        seen_terms = set()
        for word in words:
            # Filter intra-batch duplicates
            if word["term"] in seen_terms:
                dup_count += 1
                continue
            seen_terms.add(word["term"])
            
            inserted = await add_to_cart(
                DB_PATH,
                user_id,
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

        if inserted_words:
            parts.append(
                f"✅ تم إضافة <b>{len(inserted_words)}</b> كلمة جديدة "
                f"(سياق: {context_label}) لسلتك."
            )
        else:
            parts.append("ℹ️ لم يتم إضافة كلمات جديدة لسلتك.")

        if dup_count > 0:
            parts.append(
                f"ℹ️ {dup_count} كلمة تم حفظها مسبقاً في القبو الدائم (تم تخطيها)."
            )

        if term_list:
            parts.append(f"\n{term_list}")
        parts.append("\nاستمر في العمل الممتاز، دكتور. 💪")

        await _send_reply_with_retry(message, "\n".join(parts))

    except json.JSONDecodeError:
        logger.exception("Failed to parse Gemini JSON for user %d", user_id)
        await _send_reply_with_retry(message,
            "عذراً، لم أتمكن من قراءة الصورة بوضوح. "
            "يرجى إرسال صورة أوضح. 🔄"
        )

    except Exception:
        logger.exception("Gemini pipeline error for user %d", user_id)
        await _send_reply_with_retry(message,
            "عذراً، حدث خطأ أثناء معالجة الصور. "
            "يرجى المحاولة مرة أخرى. 🔄"
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

    # ── Gate: setup required ─────────────────────────────────
    if not user or not user["main_deck"]:
        await update.message.reply_text(
            "⚠️ يجب إعداد مجموعة Anki أولاً.\n"
            "أرسل /start للبدء."
        )
        return

    # ── Lazy reminder ────────────────────────────────────────
    reminder = await _check_reminder(user_id)
    await update_last_activity(DB_PATH, user_id)

    photo = update.message.photo[-1]  # highest resolution
    file_id = photo.file_id
    media_group_id = update.message.media_group_id
    caption = update.message.caption

    # ── Debounce Album Processing ────────────────────────────
    if media_group_id:
        if media_group_id not in _MEDIA_GROUPS:
            # First photo of the group
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
            # Subsequent photos in the same group
            _MEDIA_GROUPS[media_group_id]["file_ids"].append(file_id)
            if caption and not _MEDIA_GROUPS[media_group_id].get("caption"):
                _MEDIA_GROUPS[media_group_id]["caption"] = caption
        return

    # ── Single Image Processing ──────────────────────────────
    await _process_images(user_id, [file_id], update.message, reminder, context, caption)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /help & /stats Commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def help_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/help — Explain how to use the bot and custom captions."""
    text = (
        "📖 <b>دليل استخدام LingoFlow</b>\n\n"
        "1️⃣ أرسل صورة أو مجموعة صور (ألبوم) تحتوي على نص إنجليزي.\n"
        "2️⃣ سيقوم الذكاء الاصطناعي باستخراج الكلمات وترجمتها بدقة حسب السياق (طبي أو عام).\n"
        "3️⃣ الكلمات تُحفظ في <b>السلة</b> الخاصة بك.\n"
        "4️⃣ استخدم أوامر إدارة السلة: /cart و /remove و /clear.\n"
        "5️⃣ أرسل /export لتحويل الكلمات في السلة إلى ملف <code>.apkg</code> جاهز للمراجعة في Anki.\n\n"
        "💡 <b>ميزة متقدمة (المحادثة المستمرة):</b>\n"
        "يمكنك كتابة تعليق (Caption) مع الصورة لتوجيه الذكاء الاصطناعي.\n"
        "كما يمكنك إرسال رسالة نصية <b>بعد</b> استخراج الصور لتعديل النتائج.\n"
        "<i>مثال:</i> 'استخرج الأفعال فقط' أو 'أضف المزيد من الكلمات من الفقرة الثانية'."
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def stats_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/stats — Show historical export count."""
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)
    if not user:
        await update.message.reply_text("لم يتم العثور على بيانات. أرسل /start أولاً.")
        return
        
    total = user.get("total_exported", 0)
    await update.message.reply_text(
        f"📊 <b>إحصائياتك:</b>\n\n"
        f"مجموع الكلمات التي قمت بتصديرها تاريخياً: <b>{total}</b> كلمة. 🌟\n"
        f"استمر في الإنجاز!",
        parse_mode="HTML"
    )


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
        
        # Pass the text as caption/follow-up instruction
        await _process_images(user_id, file_ids, update.message, reminder, context, text)
    else:
        await update.message.reply_text(
            "أرسل صورة أولاً لأتمكن من تحليلها وتطبيق تعليماتك عليها. 📸"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /export — Generation & ConversationHandlernts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def cart_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/cart — Display a numbered list of terms in the user's cart.

    Shows term names only (no definitions) to keep the view compact.
    Prepends the lazy 24-hour reminder if applicable.
    """
    user_id = update.effective_user.id

    # ── Lazy reminder (check BEFORE updating timestamp) ──────
    reminder = await _check_reminder(user_id)
    await update_last_activity(DB_PATH, user_id)

    # ── Fetch cart ────────────────────────────────────────────
    items = await get_cart(DB_PATH, user_id)

    if not items:
        await update.message.reply_text(
            "سلتك فارغة. أرسل صورة أولاً. 📸"
        )
        return

    # ── Build numbered list ──────────────────────────────────
    term_list = "\n".join(
        f"  {i}. {item['term']}" for i, item in enumerate(items, 1)
    )

    text = (
        f"{reminder}"
        f"🛒 سلتك تحتوي على <b>{len(items)}</b> كلمة:\n\n"
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
    """/remove — Delete a specific word from the cart."""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "⚠️ يرجى تحديد الكلمة المراد حذفها.\n"
            "مثال: <code>/remove scrutinize</code>", 
            parse_mode="HTML"
        )
        return
        
    term = " ".join(context.args).strip()
    deleted = await remove_from_cart(DB_PATH, user_id, term)
    
    if deleted:
        await update.message.reply_text(
            f"🗑️ تم حذف الكلمة: <b>{term}</b> من سلتك.", 
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            f"لم يتم العثور على الكلمة: <b>{term}</b> في سلتك.", 
            parse_mode="HTML"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /clear — Clear cart with inline keyboard confirmation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Callback data constants for the inline keyboard
_CB_CLEAR_CONFIRM = "clear_confirm"
_CB_CLEAR_CANCEL = "clear_cancel"


async def clear_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/clear — Present inline keyboard to confirm cart deletion.

    Prevents accidental data loss under stress by requiring
    an explicit button press.
    """
    user_id = update.effective_user.id
    count = await get_cart_count(DB_PATH, user_id)

    if count == 0:
        await update.message.reply_text("سلتك فارغة بالفعل. ✨")
        return

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ نعم، امسح", callback_data=_CB_CLEAR_CONFIRM),
            InlineKeyboardButton("❌ لا، عد للخلف", callback_data=_CB_CLEAR_CANCEL),
        ]
    ])

    await update.message.reply_text(
        f"هل أنت متأكد من مسح <b>{count}</b> كلمة؟",
        reply_markup=keyboard,
        parse_mode="HTML",
    )


async def clear_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle the inline keyboard response for /clear confirmation."""
    query = update.callback_query
    await query.answer()  # acknowledge the button press to Telegram

    user_id = query.from_user.id

    if query.data == _CB_CLEAR_CONFIRM:
        deleted = await clear_cart(DB_PATH, user_id)
        await query.edit_message_text(
            f"🗑 تم مسح <b>{deleted}</b> كلمة من سلتك.",
            parse_mode="HTML",
        )
    elif query.data == _CB_CLEAR_CANCEL:
        await query.edit_message_text("تم الإلغاء. سلتك لم تتأثر. ✋")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  /export — Generation & ConversationHandler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def export_start(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Entry point for /export."""
    user_id = update.effective_user.id
    count = await get_cart_count(DB_PATH, user_id)

    if count == 0:
        await update.message.reply_text("سلتك فارغة. أرسل صورة أولاً. 📸")
        return ConversationHandler.END

    await update.message.reply_text(
        "ما اسم المجموعة الفرعية؟\n"
        "(أو أرسل /skip لتسمية تلقائية بتاريخ اليوم)"
    )
    return AWAITING_SUBDECK_NAME


async def skip_subdeck(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle /skip for sub-deck name."""
    sub_deck = datetime.now().strftime("Daily_%d_%b")
    return await _process_export(update, context, sub_deck)


async def receive_subdeck(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handle custom sub-deck name."""
    sub_deck = update.message.text.strip()
    
    if len(sub_deck) > 50 or "::" in sub_deck:
        await update.message.reply_text(
            "⚠️ اسم غير صالح. يرجى تجنب استخدام :: وألا يزيد عن 50 حرفاً.\n"
            "حاول مرة أخرى:"
        )
        return AWAITING_SUBDECK_NAME

    return await _process_export(update, context, sub_deck)


async def _process_export(
    update: Update, context: ContextTypes.DEFAULT_TYPE, sub_deck: str
) -> int:
    """Generate the Anki package and send it to the user."""
    user_id = update.effective_user.id
    user = await get_user(DB_PATH, user_id)
    main_deck = user["main_deck"]
    
    items = await get_cart(DB_PATH, user_id)
    
    os.makedirs("exports", exist_ok=True)
    filename = f"LingoFlow_{main_deck}_{sub_deck}.apkg".replace(" ", "_")
    output_path = os.path.join("exports", filename)
    
    status_msg = await update.message.reply_text("⏳ جاري تجهيز الملف...")
    
    try:
        generate_apkg(user_id, main_deck, sub_deck, items, output_path)
        
        with open(output_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                caption="✅ تم تصدير الملف بنجاح. بالتوفيق في مذاكرتك يا دكتور!"
            )
            
        await increment_total_exported(DB_PATH, user_id, len(items))
        await archive_cart_to_vault(DB_PATH, user_id)
        await clear_cart(DB_PATH, user_id)
        await status_msg.delete()
        
    except Exception:
        logger.exception("Export failed for user %d", user_id)
        await status_msg.edit_text("❌ حدث خطأ أثناء التصدير.")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

    return ConversationHandler.END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Application bootstrap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def post_init(application) -> None:
    """Runs after the Application is built but before polling starts.

    Initializes the database schema and discovers the best Gemini model.
    """
    global GEMINI_MODEL

    logger.info("Running post_init: initializing database...")
    await init_db(DB_PATH)

    logger.info("Running post_init: discovering Gemini model...")
    GEMINI_MODEL = _init_gemini_model()
    
    logger.info("Running post_init: setting bot commands...")
    commands = [
        BotCommand("start", "إعداد مجموعة Anki الرئيسية"),
        BotCommand("cart", "عرض محتويات السلة الحالية"),
        BotCommand("remove", "حذف كلمة معينة من السلة"),
        BotCommand("clear", "مسح جميع الكلمات من السلة"),
        BotCommand("export", "تصدير السلة إلى ملف Anki"),
        BotCommand("stats", "عرض إحصائياتك التاريخية"),
        BotCommand("help", "دليل الاستخدام والتعليمات المخصصة"),
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

    # ── /export ConversationHandler ──────────────────────────
    export_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("export", export_start)],
        states={
            AWAITING_SUBDECK_NAME: [
                CommandHandler("skip", skip_subdeck),
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_subdeck)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )

    # ── Register handlers (order matters!) ───────────────────
    application.add_handler(start_conv_handler)
    application.add_handler(export_conv_handler)
    application.add_handler(CommandHandler("cart", cart_command))
    application.add_handler(CommandHandler("remove", remove_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(CallbackQueryHandler(
        clear_callback,
        pattern=f"^({_CB_CLEAR_CONFIRM}|{_CB_CLEAR_CANCEL})$",
    ))
    
    # Text Handler MUST be registered after ConversationHandlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # ── Start polling ────────────────────────────────────────
    logger.info("LingoFlow bot starting... 🚀")
    
    while True:
        try:
            application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=())
            break
        except (TimedOut, NetworkError) as e:
            logger.warning("⚠️ Telegram servers unreachable, retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Critical error during polling: {e}")
            break
