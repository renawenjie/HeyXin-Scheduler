"""
HeyXin Daily Check-in Scheduler

Architecture:
- FastAPI web server with registration endpoint
- SQLite database for persistent user storage
- Background thread checks every minute for due check-ins
- OpenAI API (gpt-4.1-nano) extracts preferences from Dify conversation history
- Telegram Bot API sends check-in messages
- Dify API reads conversation history for preference extraction

Flow:
1. User completes onboarding in Telegram → Dify Chatflow
2. Every message triggers the Telegram Workflow, which also calls POST /register with chat_id
3. Scheduler reads the Dify conversation for that chat_id, extracts preferences via LLM
4. If onboarding is complete, user is registered/updated with their check-in schedule
5. Background job sends check-in messages at each user's preferred time

Key behavior:
- /register ALWAYS re-reads the latest conversation and re-extracts preferences
- This means if a user changes their check-in time or timezone, it auto-updates
- Rate-limited to avoid excessive LLM calls (max once per 2 minutes per user)
"""

import os
import json
import sqlite3
import logging
import threading
import time
from datetime import datetime
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────

DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "app-H8ycAHrUlDFu6YfqHpCSrBYO")
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8734162668:AAF4Fc_M-PLvBj1XdH1bDartmd9lqd_Cv08")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DB_PATH = os.environ.get("DB_PATH", "heyxin_users.db")
PORT = int(os.environ.get("PORT", "8080"))

# Rate limit: don't re-extract more than once per this many seconds per user
EXTRACT_COOLDOWN_SECONDS = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("heyxin")

# In-memory rate limiter: chat_id -> last extraction timestamp
_last_extraction: dict[str, float] = {}

# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            name TEXT,
            checkin_hour INTEGER,
            checkin_minute INTEGER,
            timezone TEXT DEFAULT 'Asia/Shanghai',
            language TEXT DEFAULT 'en',
            values_summary TEXT,
            is_active INTEGER DEFAULT 1,
            onboarding_complete INTEGER DEFAULT 0,
            last_checkin_sent TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    log.info("Database initialized: %s", DB_PATH)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_all_active_users():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE is_active = 1 AND onboarding_complete = 1 AND checkin_hour IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def get_user(chat_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
    return dict(row) if row else None


def upsert_user(chat_id: str, **kwargs):
    with get_db() as conn:
        existing = conn.execute("SELECT chat_id FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [chat_id]
            conn.execute(f"UPDATE users SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?", vals)
        else:
            kwargs["chat_id"] = chat_id
            cols = ", ".join(kwargs.keys())
            placeholders = ", ".join("?" for _ in kwargs)
            conn.execute(f"INSERT INTO users ({cols}) VALUES ({placeholders})", list(kwargs.values()))


def mark_checkin_sent(chat_id: str, date_str: str):
    with get_db() as conn:
        conn.execute("UPDATE users SET last_checkin_sent = ? WHERE chat_id = ?", (date_str, chat_id))


# ─── Dify API ────────────────────────────────────────────────────────────────

def dify_headers():
    return {"Authorization": f"Bearer {DIFY_API_KEY}"}


def get_user_conversations(chat_id: str):
    """Get conversations for a specific user (chat_id), sorted by most recently updated."""
    params = {"user": chat_id, "limit": 5, "sort_by": "-updated_at"}
    resp = requests.get(f"{DIFY_BASE_URL}/conversations", headers=dify_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_conversation_messages(conversation_id: str, user: str):
    """Get all messages from a conversation."""
    params = {"conversation_id": conversation_id, "user": user, "limit": 100}
    resp = requests.get(f"{DIFY_BASE_URL}/messages", headers=dify_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


# ─── LLM Extraction ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Analyze this conversation between a reflection bot (Xin) and a user. Extract the following:

1. name: The name the user wants to be called (not their Telegram username)
2. checkin_hour: The hour (0-23) they want daily check-ins, in their local time. Use the LATEST time they mentioned if they changed it during the conversation.
3. checkin_minute: The minute (0-59) they want daily check-ins (default 0 if not specified)
4. timezone: Their timezone in IANA format (e.g., "Asia/Shanghai", "America/New_York"). Rules:
   - If the user explicitly states a timezone (e.g., "Beijing time", "EST", "London time"), use that.
   - If the user does NOT mention any timezone at all, set timezone to "UNKNOWN".
   - Do NOT guess or assume a timezone based on language alone.
5. language: "zh" if the user primarily writes in Chinese, "en" if in English, or the appropriate ISO 639-1 code for other languages
6. values_summary: A brief summary of what matters most to them (their core values), in their language
7. onboarding_complete: true ONLY if ALL of these are present: (a) user gave their name, (b) shared their values, (c) discussed how they spend energy, (d) mentioned a check-in time. If any is missing, set to false.

IMPORTANT: Always use the MOST RECENT preferences if the user changed their mind during the conversation. For example, if they first said "8pm" then later said "10pm", use 22 for checkin_hour.

Time parsing examples: "晚上八点" = 20:00, "8pm" = 20:00, "morning 7" = 07:00, "9:30pm" = 21:30, "10 pm" = 22:00

Return ONLY valid JSON with these exact keys. No other text."""


def extract_user_preferences(messages: list) -> dict:
    """Use LLM to extract user preferences from conversation history."""
    transcript_lines = []
    for msg in messages:
        if msg.get("query"):
            transcript_lines.append(f"User: {msg['query']}")
        if msg.get("answer"):
            transcript_lines.append(f"Xin: {msg['answer']}")

    if not transcript_lines:
        return {"onboarding_complete": False}

    transcript = "\n".join(transcript_lines)

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": transcript}
            ],
            temperature=0,
            max_tokens=500,
        )
        result_text = response.choices[0].message.content.strip()
        # Clean markdown code blocks if present
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        return json.loads(result_text.strip())
    except Exception as e:
        log.error("LLM extraction failed: %s", e)
        return {"onboarding_complete": False}


# ─── Telegram API ────────────────────────────────────────────────────────────

def send_telegram_message(chat_id: str, text: str) -> bool:
    """Send a message to a Telegram user."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        log.info("Sent check-in to chat_id=%s", chat_id)
        return True
    except Exception as e:
        log.error("Failed to send to %s: %s", chat_id, e)
        return False


# ─── Check-in Messages ──────────────────────────────────────────────────────

def generate_checkin_message(user: dict) -> str:
    """Generate a personalized check-in message in the user's language."""
    name = user.get("name", "friend")
    lang = user.get("language", "en")
    values = user.get("values_summary", "")

    if lang == "zh":
        msg = f"嗨 {name}！是时候做今天的反思了 ☀️\n\n"
        if values:
            msg += f"想想你最重视的——{values}——今天你的精力花在了哪里？"
        else:
            msg += "今天你的精力花在了哪里？"
    else:
        msg = f"Hi {name}! Time for your daily reflection ☀️\n\n"
        if values:
            msg += f"Thinking about what matters most to you — {values} — how did you spend your energy today?"
        else:
            msg += "How did you spend your energy today?"

    return msg


# ─── Registration Logic ─────────────────────────────────────────────────────

def process_registration(chat_id: str) -> dict:
    """
    Called when a user sends a message. ALWAYS re-reads the latest Dify
    conversation and re-extracts preferences. This ensures changes to
    check-in time, timezone, etc. are picked up automatically.

    Rate-limited to avoid excessive LLM calls (max once per 2 minutes per user).
    """
    # Rate limiting: skip if we just extracted for this user
    now = time.time()
    last_time = _last_extraction.get(chat_id, 0)
    if now - last_time < EXTRACT_COOLDOWN_SECONDS:
        existing = get_user(chat_id)
        if existing and existing.get("onboarding_complete"):
            return {"status": "recently_checked", "user": existing}
        elif existing:
            return {"status": "recently_checked_incomplete"}
        # If no user record at all, proceed anyway
    
    _last_extraction[chat_id] = now

    # Get user's conversations from Dify (sorted by most recently updated)
    try:
        convos = get_user_conversations(chat_id)
    except Exception as e:
        log.error("Failed to get conversations for %s: %s", chat_id, e)
        return {"status": "error", "message": str(e)}

    if not convos:
        return {"status": "no_conversations"}

    # Read ALL conversations and combine messages to find the latest preferences
    # This handles the case where a user re-did onboarding in a new conversation
    all_messages = []
    for convo in convos[:3]:  # Check up to 3 most recent conversations
        try:
            messages = get_conversation_messages(convo["id"], chat_id)
            all_messages.extend(messages)
        except Exception as e:
            log.error("Failed to get messages for convo %s: %s", convo["id"], e)

    if not all_messages:
        return {"status": "no_messages"}

    # Sort messages by created_at to get chronological order
    all_messages.sort(key=lambda m: m.get("created_at", 0))

    # Extract preferences via LLM (will pick up the LATEST preferences)
    prefs = extract_user_preferences(all_messages)
    log.info("Extracted preferences for %s: %s", chat_id, json.dumps(prefs, ensure_ascii=False))

    if not prefs.get("onboarding_complete"):
        # Save partial data but don't mark as complete
        upsert_user(chat_id,
                     conversation_id=convos[0]["id"],
                     name=prefs.get("name", ""),
                     language=prefs.get("language", "en"),
                     onboarding_complete=0)
        return {"status": "onboarding_incomplete", "preferences": prefs}

    # Handle UNKNOWN timezone — use a sensible default based on language
    timezone = prefs.get("timezone", "UNKNOWN")
    if timezone == "UNKNOWN" or not timezone:
        # Default based on language as last resort
        lang = prefs.get("language", "en")
        if lang == "zh":
            timezone = "Asia/Shanghai"
        else:
            timezone = "UTC"
        log.warning("Timezone unknown for %s, defaulting to %s", chat_id, timezone)

    # Validate timezone
    try:
        ZoneInfo(timezone)
    except Exception:
        log.warning("Invalid timezone '%s' for %s, defaulting to UTC", timezone, chat_id)
        timezone = "UTC"

    # Check what changed (for logging)
    existing = get_user(chat_id)
    if existing and existing.get("onboarding_complete"):
        changes = []
        if existing.get("checkin_hour") != prefs.get("checkin_hour"):
            changes.append(f"checkin_hour: {existing.get('checkin_hour')} → {prefs.get('checkin_hour')}")
        if existing.get("checkin_minute") != prefs.get("checkin_minute", 0):
            changes.append(f"checkin_minute: {existing.get('checkin_minute')} → {prefs.get('checkin_minute', 0)}")
        if existing.get("timezone") != timezone:
            changes.append(f"timezone: {existing.get('timezone')} → {timezone}")
        if changes:
            log.info("Updating user %s: %s", chat_id, ", ".join(changes))

    # Full registration / update
    upsert_user(chat_id,
                conversation_id=convos[0]["id"],
                name=prefs.get("name", "friend"),
                checkin_hour=prefs.get("checkin_hour"),
                checkin_minute=prefs.get("checkin_minute", 0),
                timezone=timezone,
                language=prefs.get("language", "en"),
                values_summary=prefs.get("values_summary", ""),
                onboarding_complete=1)

    log.info("Registered/updated user: chat_id=%s, name=%s, checkin=%02d:%02d %s",
             chat_id, prefs.get("name"),
             prefs.get("checkin_hour", 0), prefs.get("checkin_minute", 0),
             timezone)

    return {"status": "registered", "preferences": prefs}


# ─── Background Check-in Job ────────────────────────────────────────────────

def checkin_loop():
    """Background thread that checks every 60 seconds for due check-ins."""
    log.info("Check-in loop started")
    while True:
        try:
            users = get_all_active_users()
            for user in users:
                try:
                    tz_name = user.get("timezone", "UTC")
                    try:
                        tz = ZoneInfo(tz_name)
                    except Exception:
                        tz = ZoneInfo("UTC")

                    now_user = datetime.now(tz)
                    target_hour = user.get("checkin_hour")
                    target_minute = user.get("checkin_minute", 0)

                    if target_hour is None:
                        continue

                    # Check if current time matches (within 1-minute window)
                    if now_user.hour == target_hour and now_user.minute == target_minute:
                        # Check if already sent today
                        today_str = now_user.strftime("%Y-%m-%d")
                        last_sent = user.get("last_checkin_sent", "")
                        if last_sent == today_str:
                            continue

                        # Send check-in
                        message = generate_checkin_message(user)
                        if send_telegram_message(user["chat_id"], message):
                            mark_checkin_sent(user["chat_id"], today_str)

                except Exception as e:
                    log.error("Error processing user %s: %s", user.get("chat_id"), e)

        except Exception as e:
            log.error("Check-in loop error: %s", e)

        time.sleep(60)


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="HeyXin Scheduler", version="1.1.0")


class RegisterRequest(BaseModel):
    chat_id: str


class ManualRegisterRequest(BaseModel):
    chat_id: str
    name: str
    checkin_hour: int
    checkin_minute: int = 0
    timezone: str = "Asia/Shanghai"
    language: str = "en"
    values_summary: str = ""


@app.on_event("startup")
def startup():
    init_db()
    # Start background check-in thread
    t = threading.Thread(target=checkin_loop, daemon=True)
    t.start()
    log.info("HeyXin Scheduler v1.1.0 started on port %d", PORT)


@app.get("/")
def health():
    """Health check endpoint."""
    users = get_all_active_users()
    return {
        "status": "running",
        "service": "HeyXin Scheduler",
        "version": "1.1.0",
        "active_users": len(users),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/register")
def register(req: RegisterRequest):
    """
    Register or update a user. Called by the Dify Telegram Workflow
    after each message. ALWAYS re-reads conversation and re-extracts
    preferences to pick up changes (new check-in time, timezone, etc.).
    Rate-limited to max once per 2 minutes per user.
    """
    result = process_registration(req.chat_id)
    return result


@app.post("/register/manual")
def register_manual(req: ManualRegisterRequest):
    """Manually register a user (admin endpoint for testing)."""
    upsert_user(
        req.chat_id,
        name=req.name,
        checkin_hour=req.checkin_hour,
        checkin_minute=req.checkin_minute,
        timezone=req.timezone,
        language=req.language,
        values_summary=req.values_summary,
        onboarding_complete=1,
    )
    return {"status": "registered", "chat_id": req.chat_id}


@app.get("/users")
def list_users():
    """List all registered users (admin endpoint)."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return {"users": [dict(r) for r in rows]}


@app.get("/users/{chat_id}")
def get_user_info(chat_id: str):
    """Get info for a specific user."""
    user = get_user(chat_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/users/{chat_id}/deactivate")
def deactivate_user(chat_id: str):
    """Deactivate a user (stop check-ins)."""
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active = 0 WHERE chat_id = ?", (chat_id,))
    return {"status": "deactivated", "chat_id": chat_id}


@app.post("/users/{chat_id}/activate")
def activate_user(chat_id: str):
    """Re-activate a user."""
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active = 1 WHERE chat_id = ?", (chat_id,))
    return {"status": "activated", "chat_id": chat_id}


@app.post("/checkin/test/{chat_id}")
def test_checkin(chat_id: str):
    """Send a test check-in message to a user (admin endpoint)."""
    user = get_user(chat_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    message = generate_checkin_message(user)
    success = send_telegram_message(chat_id, message)
    return {"status": "sent" if success else "failed", "message": message}


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
