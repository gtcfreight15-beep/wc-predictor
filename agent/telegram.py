"""Send a message to your own Telegram chat via the Bot API."""
import requests

from agent import config


def send_message(text: str) -> None:
    if config.DRY_RUN:
        print("----- DRY RUN: would send to Telegram -----")
        print(text)
        print("-------------------------------------------")
        return
    config.require("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=30)
    r.raise_for_status()
