import sqlite3
import os
import datetime

DB_PATH = os.environ.get('TICKETS_DB_PATH', os.path.join(os.path.dirname(__file__), 'tickets.db'))

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """
    Initializes the database and creates the tickets table if it doesn't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            actual_category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            priority TEXT NOT NULL,
            model_used TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'New',
            resolution_note TEXT DEFAULT '',
            timestamp TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            translation TEXT DEFAULT ''
        )
    """)
    conn.commit()
    
    # Run schema migrations to add columns if database already exists
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN language TEXT DEFAULT 'en'")
        conn.commit()
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN translation TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
        
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translation_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_text TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            transliterated_text TEXT DEFAULT '',
            timestamp TEXT NOT NULL,
            UNIQUE(source_text, source_lang, target_lang)
        )
    """)
    conn.commit()

    conn.close()
    
    # Seed sample multi-lingual ticket data if queue is empty
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]
    if count == 0:
        import datetime
        now = datetime.datetime.now()
        seeds = [
            (
                "मेरा खाता ब्लॉक हो गया है और मैं लॉग इन नहीं कर पा रहा हूँ, कृपया तुरंत मदद करें।",
                "Account Access", "Account Access", "Negative", "High", "logistic", "New", "",
                (now - datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "hi", "My account has been blocked and I cannot log in, please help immediately."
            ),
            (
                "I was double charged on my credit card for the annual subscription renewal fees. This is unauthorized and fraudulent!",
                "Billing Issue", "Billing Issue", "Negative", "High", "svc", "In Progress", "Agent is currently inspecting payment processor logs.",
                (now - datetime.timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
                "en", ""
            ),
            (
                "ನನ್ನ ಮೊಬೈಲ್ ಆಪ್‌ನಲ್ಲಿ ಅಪ್‌ಡೇಟ್ ಆದ ನಂತರ ಡೇಟಾ ಸಿಂಕ್ ಆಗ್ತಿಲ್ಲ.",
                "Technical Issue", "Technical Issue", "Negative", "Medium", "naive_bayes", "New", "",
                (now - datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "kn", "Data is not syncing in my mobile app after the update."
            ),
            (
                "Hello, I would like to know if your system supports custom dark mode theme settings or background options?",
                "General Inquiry", "General Inquiry", "Positive", "Low", "logistic", "Resolved", "Sent styling options guide and documentation link.",
                (now - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "en", ""
            ),
            (
                "I requested a refund for my cancelled subscription last week but haven't received the credit yet.",
                "Refund Request", "Refund Request", "Neutral", "Medium", "svc", "New", "",
                (now - datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "en", ""
            )
        ]
        cursor.executemany("""
            INSERT INTO tickets (transcript, predicted_category, actual_category, sentiment, priority, model_used, status, resolution_note, timestamp, language, translation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seeds)
        conn.commit()
    conn.close()

def add_ticket(transcript, category, sentiment, priority, model_used, language='en', translation=''):
    """
    Adds a new support ticket to the queue database.
    """
    # Ensure database is initialized
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO tickets (transcript, predicted_category, actual_category, sentiment, priority, model_used, status, timestamp, language, translation)
        VALUES (?, ?, ?, ?, ?, ?, 'New', ?, ?, ?)
    """, (transcript, category, category, sentiment, priority, model_used, timestamp, language, translation))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_all_tickets():
    """
    Retrieves all tickets from the database, sorted by priority (High, then Medium, then Low) and timestamp descending.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    # Order: High first, then Medium, then Low. Custom order using CASE
    cursor.execute("""
        SELECT id, transcript, predicted_category, actual_category, sentiment, priority, model_used, status, resolution_note, timestamp, language, translation
        FROM tickets
        ORDER BY 
            CASE priority 
                WHEN 'High' THEN 1 
                WHEN 'Medium' THEN 2 
                WHEN 'Low' THEN 3 
                ELSE 4 
            END ASC,
            timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    tickets = []
    for r in rows:
        tickets.append({
            'id': r[0],
            'transcript': r[1],
            'predicted_category': r[2],
            'actual_category': r[3],
            'sentiment': r[4],
            'priority': r[5],
            'model_used': r[6],
            'status': r[7],
            'resolution_note': r[8] if r[8] else '',
            'timestamp': r[9],
            'language': r[10] if len(r) > 10 and r[10] else 'en',
            'translation': r[11] if len(r) > 11 and r[11] else ''
        })
    return tickets

def update_ticket(ticket_id, category, priority, status, resolution_note):
    """
    Updates the fields of a ticket (typically by an administrator).
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE tickets
        SET actual_category = ?, priority = ?, status = ?, resolution_note = ?
        WHERE id = ?
    """, (category, priority, status, resolution_note, ticket_id))
    conn.commit()
    conn.close()

def delete_ticket(ticket_id):
    """
    Deletes a ticket from the database.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
    conn.commit()
    conn.close()

def clear_all_tickets():
    """
    Clears all records from the tickets table.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets")
    conn.commit()
    conn.close()

def get_cached_translation(text, source_lang, target_lang):
    """
    Looks up a translation and optional transliteration in the SQLite cache.
    """
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT translated_text, transliterated_text FROM translation_cache
            WHERE source_text = ? AND source_lang = ? AND target_lang = ?
        """, (text, source_lang, target_lang))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "translated_text": row[0],
                "transliterated_text": row[1] if row[1] else ""
            }
    except sqlite3.OperationalError:
        pass
    return None

def set_cached_translation(text, source_lang, target_lang, translated_text, transliterated_text=""):
    """
    Saves or updates a translation and transliteration in the SQLite cache.
    """
    init_db()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT OR REPLACE INTO translation_cache 
            (source_text, source_lang, target_lang, translated_text, transliterated_text, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (text, source_lang, target_lang, translated_text, transliterated_text, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to write translation cache: {e}")
