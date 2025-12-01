#!/usr/bin/env python3
"""
Migration script for adding provisional_tokens to users table
"""

import sqlite3
from pathlib import Path

def migrate_database():
    """Add provisional_tokens column to users table"""
    db_path = Path("data/sheily_dashboard.db")

    if not db_path.exists():
        print("⚠️ Database file not found. Creating new database...")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if provisional_tokens column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if 'provisional_tokens' not in column_names:
            print("🔄 Adding provisional_tokens column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN provisional_tokens INTEGER DEFAULT 0")
            print("✅ provisional_tokens column added")
        else:
            print("✅ provisional_tokens column already exists")

        # Create hack_memori_rewards table for tracking rewards per session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hack_memori_rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT NOT NULL,
                qa_count INTEGER DEFAULT 0,
                tokens_earned INTEGER DEFAULT 0,
                training_triggered BOOLEAN DEFAULT FALSE,
                training_completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        print("✅ hack_memori_rewards table created")

        conn.commit()
        conn.close()
        print("✅ Database migration completed successfully")

    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_database()
