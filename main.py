"""
LingoFlow — Streamlit Cloud Entrypoint
======================================
Provides a simple Streamlit web UI to satisfy cloud health checks,
and concurrently starts the Telegram bot in a background thread.
"""

import threading
import streamlit as st
import bot

# ── Streamlit UI ──
st.set_page_config(page_title="LingoFlow Bot", page_icon="🤖")
st.title("LingoFlow Bot 🤖")
st.write("✅ Bot is actively running in the background.")
st.write("This page keeps the Streamlit Cloud instance alive.")

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
