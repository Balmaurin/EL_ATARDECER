#!/usr/bin/env python3
"""
Reset Tokens Script - Sheily AI Dashboard
=========================================

Resetea todos los tokens y tokens provisionales a cero.
"""

import sqlite3
import os
from pathlib import Path

def reset_tokens():
    """Reset all tokens to zero"""
    try:
        # Database path
        db_path = Path("data/sheily_dashboard.db")

        if not db_path.exists():
            print(f"❌ Database not found: {db_path}")
            return False

        with sqlite3.connect(db_path) as conn:
            # Reset tokens for all users
            conn.execute("UPDATE users SET tokens = 0, provisional_tokens = 0")

            # Commit changes
            conn.commit()

            print("✅ Tokens reset successfully!")
            print("   - confirmed_tokens: 0")
            print("   - provisional_tokens: 0")

            return True

    except Exception as e:
        print(f"❌ Error resetting tokens: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Resetting all tokens...")
    success = reset_tokens()

    if success:
        print("\n🎯 Reset complete! Start fresh with token system.")
    else:
        print("\n❌ Reset failed!")
