"""
Agent Learning - REAL Implementation
Persists learning experiences to database for continuous improvement
"""
import logging
import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """Learning experience data structure"""
    experience_id: str
    agent_id: str
    task_type: str
    outcome: str
    metrics: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metrics'] = json.dumps(self.metrics)  # Store as JSON string
        return data


class LearningDatabase:
    """REAL database for storing learning experiences"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Use default path in data directory
            base_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "learning"
            base_path.mkdir(parents=True, exist_ok=True)
            db_path = str(base_path / "agent_learning.db")
        
        self.db_path = db_path
        self._init_database()
        logger.info(f"✅ LearningDatabase initialized at {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_experiences (
                    experience_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_id 
                ON learning_experiences(agent_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type 
                ON learning_experiences(task_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON learning_experiences(timestamp)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def store_experience(self, experience: LearningExperience) -> bool:
        """REAL storage of learning experience"""
        try:
            data = experience.to_dict()
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_experiences 
                    (experience_id, agent_id, task_type, outcome, metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    data['experience_id'],
                    data['agent_id'],
                    data['task_type'],
                    data['outcome'],
                    data['metrics'],
                    data['timestamp']
                ))
                conn.commit()
            logger.debug(f"Stored learning experience {experience.experience_id} for agent {experience.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store learning experience: {e}")
            return False
    
    def get_agent_experiences(
        self, 
        agent_id: str, 
        limit: int = 100,
        task_type: Optional[str] = None
    ) -> List[LearningExperience]:
        """Get learning experiences for an agent"""
        try:
            with self._get_connection() as conn:
                if task_type:
                    cursor = conn.execute("""
                        SELECT * FROM learning_experiences
                        WHERE agent_id = ? AND task_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (agent_id, task_type, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM learning_experiences
                        WHERE agent_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (agent_id, limit))
                
                experiences = []
                for row in cursor.fetchall():
                    metrics = json.loads(row['metrics'])
                    experiences.append(LearningExperience(
                        experience_id=row['experience_id'],
                        agent_id=row['agent_id'],
                        task_type=row['task_type'],
                        outcome=row['outcome'],
                        metrics=metrics,
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    ))
                return experiences
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    def get_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get learning statistics"""
        try:
            with self._get_connection() as conn:
                if agent_id:
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(DISTINCT task_type) as task_types,
                            AVG(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_rate
                        FROM learning_experiences
                        WHERE agent_id = ?
                    """, (agent_id,))
                else:
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(DISTINCT agent_id) as agents,
                            COUNT(DISTINCT task_type) as task_types,
                            AVG(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_rate
                        FROM learning_experiences
                    """)
                
                row = cursor.fetchone()
                return dict(row)
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# Global database instance
_learning_db: Optional[LearningDatabase] = None


def get_learning_database() -> LearningDatabase:
    """Get global learning database instance"""
    global _learning_db
    if _learning_db is None:
        _learning_db = LearningDatabase()
    return _learning_db


def record_agent_experience(experience: LearningExperience) -> bool:
    """
    REAL implementation - Record agent learning experience to database
    """
    db = get_learning_database()
    return db.store_experience(experience)
