"""
Gamified Database - Sistema de base de datos para gamificación educativa
"""
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class GamifiedDatabase:
    """Database manager for educational gamification system"""
    
    def __init__(self, db_path: str = "data/gamification.db"):
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    total_points INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    experience INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Achievements table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS achievements (
                    achievement_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    achievement_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    points_awarded INTEGER DEFAULT 0,
                    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Challenges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    challenge_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    difficulty TEXT,
                    points_reward INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            # User challenges (participation)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    challenge_id TEXT NOT NULL,
                    status TEXT DEFAULT 'in_progress',
                    progress REAL DEFAULT 0.0,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (challenge_id) REFERENCES challenges(challenge_id)
                )
            """)
            
            # Leaderboard
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leaderboard (
                    rank INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    total_score INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Badges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS badges (
                    badge_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    icon TEXT,
                    rarity TEXT DEFAULT 'common',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User badges
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_badges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    badge_id TEXT NOT NULL,
                    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (badge_id) REFERENCES badges(badge_id)
                )
            """)
            
            conn.commit()
            logger.info(f"Gamified database initialized at {self.db_path}")
    
    # ==================== USER OPERATIONS ====================
    
    def create_user(self, user_id: str, username: str) -> bool:
        """Create new user"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (user_id, username) VALUES (?, ?)",
                    (user_id, username)
                )
                conn.commit()
                logger.info(f"Created user: {username}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"User {username} already exists")
            return False
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """Get user statistics"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_user_points(self, user_id: str, points: int) -> bool:
        """Update user points and level"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # Update points
            cursor.execute(
                "UPDATE users SET total_points = total_points + ?, experience = experience + ? WHERE user_id = ?",
                (points, points, user_id)
            )
            
            updated = cursor.rowcount > 0
            
            # Check for level up (every 1000 XP = 1 level)
            cursor.execute("SELECT experience, level FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                experience, current_level = row
                new_level = 1 + (experience // 1000)
                if new_level > current_level:
                    cursor.execute("UPDATE users SET level = ? WHERE user_id = ?", (new_level, user_id))
                    logger.info(f"User {user_id} leveled up to {new_level}!")
            
            conn.commit()
            return updated
    
    def get_all_users(self, limit: int = 100) -> List[Dict]:
        """Get all users"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY total_points DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== ACHIEVEMENT OPERATIONS ====================
    
    def award_achievement(self, user_id: str, achievement_data: Dict) -> str:
        """Award achievement to user"""
        achievement_id = f"ach_{user_id}_{int(datetime.now().timestamp())}"
        points = achievement_data.get('points', 0)
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO achievements 
                (achievement_id, user_id, achievement_type, title, description, points_awarded)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                achievement_id,
                user_id,
                achievement_data.get('type', 'general'),
                achievement_data['title'],
                achievement_data.get('description', ''),
                points
            ))
            
            # Update user points directly in same connection
            cursor.execute(
                "UPDATE users SET total_points = total_points + ?, experience = experience + ? WHERE user_id = ?",
                (points, points, user_id)
            )
            
            # Check for level up
            cursor.execute("SELECT experience, level FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                experience, current_level = row
                new_level = 1 + (experience // 1000)
                if new_level > current_level:
                    cursor.execute("UPDATE users SET level = ? WHERE user_id = ?", (new_level, user_id))
                    logger.info(f"User {user_id} leveled up to {new_level}!")
            
            conn.commit()
        
        logger.info(f"Awarded achievement '{achievement_data['title']}' to user {user_id}")
        return achievement_id
    
    def get_user_achievements(self, user_id: str) -> List[Dict]:
        """Get all achievements for user"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM achievements WHERE user_id = ? ORDER BY earned_at DESC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== CHALLENGE OPERATIONS ====================
    
    def create_challenge(self, challenge_data: Dict) -> str:
        """Create new challenge"""
        challenge_id = f"chal_{int(datetime.now().timestamp())}"
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO challenges 
                (challenge_id, title, description, difficulty, points_reward, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                challenge_id,
                challenge_data['title'],
                challenge_data.get('description', ''),
                challenge_data.get('difficulty', 'medium'),
                challenge_data.get('points_reward', 100),
                challenge_data.get('expires_at')
            ))
            conn.commit()
        
        logger.info(f"Created challenge: {challenge_data['title']}")
        return challenge_id
    
    def get_active_challenges(self) -> List[Dict]:
        """Get all active challenges"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM challenges WHERE status = 'active' ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def join_challenge(self, user_id: str, challenge_id: str) -> bool:
        """User joins a challenge"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_challenges (user_id, challenge_id, status, progress)
                    VALUES (?, ?, 'in_progress', 0.0)
                """, (user_id, challenge_id))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"User {user_id} already joined challenge {challenge_id}")
            return False
    
    def update_challenge_progress(self, user_id: str, challenge_id: str, progress: float) -> bool:
        """Update user's progress in challenge"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # Update progress
            cursor.execute("""
                UPDATE user_challenges 
                SET progress = ?, status = CASE WHEN ? >= 1.0 THEN 'completed' ELSE 'in_progress' END,
                    completed_at = CASE WHEN ? >= 1.0 THEN CURRENT_TIMESTAMP ELSE NULL END
                WHERE user_id = ? AND challenge_id = ?
            """, (progress, progress, progress, user_id, challenge_id))
            
            # If completed, award points
            if progress >= 1.0:
                cursor.execute("SELECT points_reward FROM challenges WHERE challenge_id = ?", (challenge_id,))
                row = cursor.fetchone()
                if row:
                    points = row[0]
                    self.update_user_points(user_id, points)
                    logger.info(f"User {user_id} completed challenge {challenge_id}, awarded {points} points")
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_user_challenges(self, user_id: str) -> List[Dict]:
        """Get user's challenges"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT uc.*, c.title, c.description, c.difficulty, c.points_reward
                FROM user_challenges uc
                JOIN challenges c ON uc.challenge_id = c.challenge_id
                WHERE uc.user_id = ?
                ORDER BY uc.id DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== LEADERBOARD OPERATIONS ====================
    
    def update_leaderboard(self):
        """Update leaderboard rankings"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # Clear current leaderboard
            cursor.execute("DELETE FROM leaderboard")
            
            # Rebuild from user scores
            cursor.execute("""
                INSERT INTO leaderboard (rank, user_id, total_score)
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY total_points DESC) as rank,
                    user_id,
                    total_points
                FROM users
                ORDER BY total_points DESC
            """)
            
            conn.commit()
    
    def get_leaderboard(self, limit: int = 100) -> List[Dict]:
        """Get top users from leaderboard"""
        self.update_leaderboard()
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT l.rank, l.user_id, u.username, l.total_score, u.level
                FROM leaderboard l
                JOIN users u ON l.user_id = u.user_id
                ORDER BY l.rank
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_user_rank(self, user_id: str) -> Optional[int]:
        """Get user's rank in leaderboard"""
        self.update_leaderboard()
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rank FROM leaderboard WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    # ==================== BADGE OPERATIONS ====================
    
    def create_badge(self, badge_data: Dict) -> str:
        """Create new badge"""
        badge_id = f"badge_{int(datetime.now().timestamp())}"
        
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO badges (badge_id, name, description, icon, rarity)
                VALUES (?, ?, ?, ?, ?)
            """, (
                badge_id,
                badge_data['name'],
                badge_data.get('description', ''),
                badge_data.get('icon', '🏆'),
                badge_data.get('rarity', 'common')
            ))
            conn.commit()
        
        return badge_id
    
    def award_badge(self, user_id: str, badge_id: str) -> bool:
        """Award badge to user"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_badges (user_id, badge_id)
                    VALUES (?, ?)
                """, (user_id, badge_id))
                conn.commit()
                logger.info(f"Awarded badge {badge_id} to user {user_id}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"User {user_id} already has badge {badge_id}")
            return False
    
    def get_user_badges(self, user_id: str) -> List[Dict]:
        """Get user's badges"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT b.*, ub.earned_at
                FROM user_badges ub
                JOIN badges b ON ub.badge_id = b.badge_id
                WHERE ub.user_id = ?
                ORDER BY ub.earned_at DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== STATISTICS ====================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total users
            cursor.execute("SELECT COUNT(*) FROM users")
            stats['total_users'] = cursor.fetchone()[0]
            
            # Total achievements awarded
            cursor.execute("SELECT COUNT(*) FROM achievements")
            stats['total_achievements'] = cursor.fetchone()[0]
            
            # Active challenges
            cursor.execute("SELECT COUNT(*) FROM challenges WHERE status = 'active'")
            stats['active_challenges'] = cursor.fetchone()[0]
            
            # Total points distributed
            cursor.execute("SELECT SUM(total_points) FROM users")
            stats['total_points_distributed'] = cursor.fetchone()[0] or 0
            
            # Average user level
            cursor.execute("SELECT AVG(level) FROM users")
            stats['average_user_level'] = round(cursor.fetchone()[0] or 1, 2)
            
            return stats


# Global instance
_gamified_db = None

def get_gamified_database(db_path: str = "data/gamification.db") -> GamifiedDatabase:
    """Get global gamified database instance"""
    global _gamified_db
    if _gamified_db is None:
        _gamified_db = GamifiedDatabase(db_path)
    return _gamified_db
