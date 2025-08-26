#This script is being used to handle all DB operations for chat history.
#It stores chat history in a SQLite database with a schema that supports multiple chat threads, each identified by a unique thread_id.


import sqlite3
import json
from datetime import datetime
from pathlib import Path

# =========================================================================
# === NOTE: This assumes a local database. For Streamlit Cloud persistence,
# ===       you MUST use an external service like Firebase.
# =========================================================================

DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB_PATH = DB_DIR / "chat_history.db"

def init_db():
    """Initialize the SQLite DB with a chat_history table."""
    # Ensure the directory exists
    if not DB_DIR.exists():
        DB_DIR.mkdir()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            mode TEXT NOT NULL,
            method TEXT,
            verification TEXT,
            confidence REAL,
            source TEXT,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON chat_history (thread_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON chat_history (user_id)")
    conn.commit()
    conn.close()


def save_chat(user_id, thread_id, title, query, answer, mode, response_data, response_time):
    """Save a chat interaction with a user_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Use json.dumps for the entire response_data for a more robust schema
    # The timestamp is now handled by the database's CURRENT_TIMESTAMP default
    cursor.execute("""
        INSERT INTO chat_history
        (thread_id, user_id, title, query, answer, mode, method, verification, confidence, source, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        thread_id,
        user_id,
        title,
        query,
        answer,
        mode,
        response_data.get("method"),
        response_data.get("verification"),
        response_data.get("confidence"),
        response_data.get("source"),
        response_time
    ))

    conn.commit()
    conn.close()


def update_chat_title(user_id, thread_id, new_title):
    """Updates the title for a given chat thread for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE chat_history
        SET title = ?
        WHERE thread_id = ? AND user_id = ?
    """, (new_title, thread_id, user_id))
    conn.commit()
    conn.close()


def load_chats(user_id, limit=20, thread_id=None):
    """
    Load recent chat conversations for a user, or a specific thread if thread_id is provided.
    This function is now capable of handling both use cases.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    cursor = conn.cursor()

    # If a specific thread_id is provided, load all messages for that thread
    if thread_id:
        cursor.execute("""
            SELECT query, answer, thread_id, title FROM chat_history 
            WHERE user_id = ? AND thread_id = ? 
            ORDER BY timestamp ASC
        """, (user_id, thread_id))
        rows = cursor.fetchall()
        
        if rows:
            conversation_data = {
                "thread_id": rows[0]["thread_id"],
                "title": rows[0]["title"],
                "messages": [
                    {"query": row["query"], "answer": row["answer"]}
                    for row in rows
                ]
            }
            conn.close()
            return [conversation_data]
        else:
            conn.close()
            return []

    # If no thread_id, load the latest threads for the sidebar
    cursor.execute("""
        SELECT thread_id, title, user_id, timestamp FROM chat_history 
        WHERE user_id = ? 
        GROUP BY thread_id 
        ORDER BY MAX(timestamp) DESC 
        LIMIT ?
    """, (user_id, limit))
    
    threads = cursor.fetchall()
    
    conversations = []
    for thread in threads:
        thread_id = thread["thread_id"]
        title = thread["title"]
        
        # Load the full chat data for each thread (still an N+1 query, but functional)
        cursor.execute("""
            SELECT query, answer FROM chat_history 
            WHERE thread_id = ? AND user_id = ?
            ORDER BY timestamp ASC
        """, (thread_id, user_id))
        messages_rows = cursor.fetchall()
        
        messages = [{"query": msg["query"], "answer": msg["answer"]} for msg in messages_rows]
        
        chat_title = title if title else messages[0]["query"]
        conversations.append({
            "thread_id": thread_id,
            "title": chat_title,
            "messages": messages,
            "user_id": user_id
        })
            
    conn.close()
    return conversations

def migrate_schema():
    """Add user_id column to existing database and migrate old data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user_id column exists
        cursor.execute("PRAGMA table_info(chat_history)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "user_id" not in columns:
            print("Migrating database: adding user_id column...")
            cursor.execute("ALTER TABLE chat_history ADD COLUMN user_id TEXT DEFAULT 'legacy'")
            conn.commit()
            print("Migration completed! Existing chats marked as 'legacy'")
        else:
            print("Database already has user_id column.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

def load_latest_chat(user_id):
    """Loads the most recent chat from the database for the given user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # === CORRECTED: changed "chats" to "chat_history" ===
    c.execute(
        """SELECT thread_id FROM chat_history WHERE user_id = ? 
        ORDER BY timestamp DESC LIMIT 1""",
        (user_id,)
    )
    # ====================================================
    latest_thread = c.fetchone()
    
    if latest_thread:
        latest_thread_id = latest_thread[0]
        # === Using the corrected load_chats function to get the full conversation ===
        conversation = load_chats(user_id=user_id, thread_id=latest_thread_id)
        conn.close()
        return conversation[0] if conversation else None
    
    conn.close()
    return None



# import sqlite3
# from datetime import datetime
# import json

# DB_PATH = "chat_history.db"

# def init_db():
#     """Initialize the SQLite DB with a fresh chat_history table."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     # Drop the existing table to start fresh with the new schema
#     #cursor.execute("DROP TABLE IF EXISTS chat_history")
    
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS chat_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             thread_id TEXT NOT NULL,
#             title TEXT,
#             query TEXT NOT NULL,
#             answer TEXT NOT NULL,
#             mode TEXT NOT NULL,
#             method TEXT,
#             verification TEXT,
#             confidence REAL,
#             source TEXT,
#             response_time REAL,
#             timestamp TEXT DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON chat_history (thread_id)")
#     conn.commit()
#     conn.close()


# def save_chat(thread_id, title, query, answer, mode, response_data, response_time):
#     """Save a single chat interaction to the DB, linked by a thread_id."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     cursor.execute("""
#         INSERT INTO chat_history
#         (thread_id, title, query, answer, mode, method, verification, confidence, source, response_time, timestamp)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         thread_id,
#         title,
#         query,
#         answer,
#         mode,
#         response_data.get("method"),
#         response_data.get("verification"),
#         response_data.get("confidence"),
#         response_data.get("source"),
#         response_time,
#         datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     ))

#     conn.commit()
#     conn.close()


# def update_chat_title(thread_id, new_title):
#     """Updates the title for a given chat thread."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         UPDATE chat_history
#         SET title = ?
#         WHERE thread_id = ?
#     """, (new_title, thread_id))
#     conn.commit()
#     conn.close()


# def load_chats(limit=20):
#     """Load the most recent chat conversations, grouped by thread_id."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     cursor.execute("""
#         SELECT DISTINCT thread_id, title FROM chat_history
#         ORDER BY timestamp DESC
#         LIMIT ?
#     """, (limit,))
    
#     thread_info = cursor.fetchall()
    
#     conversations = []
#     for thread_id, title in thread_info:
#         cursor.execute("""
#             SELECT query, answer, mode, method, verification, confidence, source, response_time, timestamp
#             FROM chat_history
#             WHERE thread_id = ?
#             ORDER BY timestamp ASC
#         """, (thread_id,))
        
#         messages = []
#         for row in cursor.fetchall():
#             messages.append({
#                 "query": row[0],
#                 "answer": row[1],
#                 "mode": row[2],
#                 "method": row[3],
#                 "verification": row[4],
#                 "confidence": row[5],
#                 "source": row[6],
#                 "response_time": row[7],
#                 "timestamp": row[8],
#             })
            
#         if messages:
#             chat_title = title if title else messages[0]["query"]
#             conversations.append({
#                 "thread_id": thread_id,
#                 "title": chat_title,
#                 "messages": messages
#             })
            
#     conn.close()
#     return conversations