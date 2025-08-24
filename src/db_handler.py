#This script is being used to handle all DB operations for chat history.
#It stores chat history in a SQLite database with a schema that supports multiple chat threads, each identified by a unique thread_id.






import sqlite3
from datetime import datetime
import json

DB_PATH = "chat_history.db"

def init_db():
    """Initialize the SQLite DB with a fresh chat_history table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Drop the existing table to start fresh with the new schema
    #cursor.execute("DROP TABLE IF EXISTS chat_history")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            title TEXT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            mode TEXT NOT NULL,
            method TEXT,
            verification TEXT,
            confidence REAL,
            source TEXT,
            response_time REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON chat_history (thread_id)")
    conn.commit()
    conn.close()


def save_chat(thread_id, title, query, answer, mode, response_data, response_time):
    """Save a single chat interaction to the DB, linked by a thread_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history
        (thread_id, title, query, answer, mode, method, verification, confidence, source, response_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        thread_id,
        title,
        query,
        answer,
        mode,
        response_data.get("method"),
        response_data.get("verification"),
        response_data.get("confidence"),
        response_data.get("source"),
        response_time,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def update_chat_title(thread_id, new_title):
    """Updates the title for a given chat thread."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE chat_history
        SET title = ?
        WHERE thread_id = ?
    """, (new_title, thread_id))
    conn.commit()
    conn.close()


def load_chats(limit=20):
    """Load the most recent chat conversations, grouped by thread_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT thread_id, title FROM chat_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    thread_info = cursor.fetchall()
    
    conversations = []
    for thread_id, title in thread_info:
        cursor.execute("""
            SELECT query, answer, mode, method, verification, confidence, source, response_time, timestamp
            FROM chat_history
            WHERE thread_id = ?
            ORDER BY timestamp ASC
        """, (thread_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "query": row[0],
                "answer": row[1],
                "mode": row[2],
                "method": row[3],
                "verification": row[4],
                "confidence": row[5],
                "source": row[6],
                "response_time": row[7],
                "timestamp": row[8],
            })
            
        if messages:
            chat_title = title if title else messages[0]["query"]
            conversations.append({
                "thread_id": thread_id,
                "title": chat_title,
                "messages": messages
            })
            
    conn.close()
    return conversations