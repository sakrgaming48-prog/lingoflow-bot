"""
LingoFlow — Streamlit Cloud Entrypoint
======================================
Provides a simple Streamlit web UI to satisfy cloud health checks,
and concurrently starts the Telegram bot in a background thread.
Includes robust asyncio event loop initialization for Python 3.12+.
"""

import asyncio
import threading
import streamlit as st
import bot

# ── Ensure Event Loop Exists (Python 3.12+ Compat) ──
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ── Streamlit UI ──
st.set_page_config(page_title="LingoFlow Bot", page_icon="🩺")
st.title("🩺 LingoFlow Bot is Active")
st.success("✅ The Telegram bot is actively polling in the background.")
st.info("This Streamlit page ensures the cloud instance stays alive.")

# ── Background Bot Thread ──
@st.cache_resource
def start_bot_thread():
    print("Starting Telegram bot in a background thread...")
    # Start the Telegram bot in a background daemon thread
    thread = threading.Thread(target=bot.main, daemon=True)
    thread.start()
    return thread

# Start the bot thread only once per Streamlit instance
start_bot_thread()
