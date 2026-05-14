"""
LingoFlow — Streamlit Cloud Entrypoint
======================================
Provides a simple Streamlit web UI to satisfy cloud health checks,
and concurrently starts the Telegram bot in a background thread.
"""

import asyncio
import threading
import streamlit as st
import bot

# ── Streamlit UI ──
st.set_page_config(page_title="LingoFlow Bot", page_icon="🩺")
st.title("🩺 LingoFlow Bot is Active")
st.success("✅ The Telegram bot is actively polling in the background.")
st.info("This Streamlit page ensures the cloud instance stays alive.")

# ── Background Bot Thread ──
@st.cache_resource
def start_bot_thread():
    print("Starting Telegram bot in a background thread...")
    
    def run_bot():
        # Ensure this new thread has its own asyncio event loop
        asyncio.set_event_loop(asyncio.new_event_loop())
        bot.main()

    # Start the Telegram bot in a background daemon thread
    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()
    return thread

# Start the bot thread only once per Streamlit instance
start_bot_thread()
