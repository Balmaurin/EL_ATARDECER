"""
Agent Quality Evaluation Module - REAL Implementation
Comprehensive quality evaluation based on actual agent performance
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from .agent_learning import get_learning_database, LearningExperience

logger = logging.getLogger(__name__)


class AgentQualityEvaluator:
    """REAL quality evaluator based on actual performance data"""
    
    def __init__(self):
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=5)
        logger.info("✅ AgentQualityEvaluator initialized")
    
    def evaluate_agent_quality(
        self, 
        agent_id: str, 
        lookback_days: int = 7,
        min_experiences: int = 5
    ) -> Dict[str, Any]:
        """
        REAL quality evaluation based on actual learning experiences
        
        Args:
            agent_id: Agent identifier
            lookback_days: Number of days to look back for evaluation
            min_experiences: Minimum experiences required for reliable evaluation
            
        Returns:
            Dictionary with real quality metrics
        """
        # Check cache
        cache_key = f"{agent_id}_{lookback_days}"
        if cache_key in self.evaluation_cache:
            cached = self.evaluation_cache[cache_key]
            if (datetime.now() - cached['evaluated_at']) < self.cache_ttl:
                return cached['metrics']
        
        try:
            db = get_learning_database()
            
            # Get recent experiences
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            all_experiences = db.get_agent_experiences(agent_id, limit=1000)
            recent_experiences = [
                exp for exp in all_experiences
                if exp.timestamp >= cutoff_date
            ]
            
            if len(recent_experiences) < min_experiences:
                logger.warning(
                    f"Insufficient experiences for agent {agent_id}: "
                    f"{len(recent_experiences)} < {min_experiences}"
                )
                return {
                    "quality_score": 0.5,  # Neutral when insufficient data
                    "status": "insufficient_data",
                    "agent_id": agent_id,
                    "metrics": {
                        "performance": 0.5,
                        "reliability": 0.5,
                        "efficiency": 0.5,
                        "experience_count": len(recent_experiences)
                    },
                    "warning": f"Only {len(recent_experiences)} experiences available"
                }
            
            # Calculate real metrics
            total_experiences = len(recent_experiences)
            success_count = sum(1 for exp in recent_experiences if exp.outcome == "success")
            success_rate = success_count / total_experiences if total_experiences > 0 else 0.0
            
            # Calculate performance from metrics
            performance_scores = []
            reliability_scores = []
            efficiency_scores = []
            
            for exp in recent_experiences:
                metrics = exp.metrics
                if 'performance' in metrics:
                    performance_scores.append(float(metrics['performance']))
                if 'reliability' in metrics:
                    reliability_scores.append(float(metrics['reliability']))
                if 'efficiency' in metrics:
                    efficiency_scores.append(float(metrics['efficiency']))
                elif 'execution_time' in metrics and 'expected_time' in metrics:
                    # Calculate efficiency from execution time
                    exec_time = float(metrics.get('execution_time', 1.0))
                    expected_time = float(metrics.get('expected_time', 1.0))
                    efficiency = min(1.0, expected_time / exec_time) if exec_time > 0 else 0.0
                    efficiency_scores.append(efficiency)
            
            # Calculate averages
            avg_performance = (
                sum(performance_scores) / len(performance_scores)
                if performance_scores else success_rate
            )
            avg_reliability = (
                sum(reliability_scores) / len(reliability_scores)
                if reliability_scores else success_rate
            )
            avg_efficiency = (
                sum(efficiency_scores) / len(efficiency_scores)
                if efficiency_scores else 0.85
            )
            
            # Overall quality score (weighted average)
            quality_score = (
                avg_performance * 0.4 +
                avg_reliability * 0.4 +
                avg_efficiency * 0.2
            )
            
            # Determine status
            if quality_score >= 0.9:
                status = "excellent"
            elif quality_score >= 0.75:
                status = "good"
            elif quality_score >= 0.6:
                status = "acceptable"
            else:
                status = "needs_improvement"
            
            result = {
                "quality_score": round(quality_score, 3),
                "status": status,
                "agent_id": agent_id,
                "metrics": {
                    "performance": round(avg_performance, 3),
                    "reliability": round(avg_reliability, 3),
                    "efficiency": round(avg_efficiency, 3),
                    "success_rate": round(success_rate, 3),
                    "total_experiences": total_experiences,
                    "success_count": success_count,
                },
                "evaluated_at": datetime.now().isoformat(),
                "lookback_days": lookback_days,
            }
            
            # Cache result
            self.evaluation_cache[cache_key] = {
                'metrics': result,
                'evaluated_at': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating agent quality for {agent_id}: {e}", exc_info=True)
            return {
                "quality_score": 0.0,
                "status": "error",
                "agent_id": agent_id,
                "error": str(e),
                "metrics": {}
            }


# Global evaluator instance
_quality_evaluator: Optional[AgentQualityEvaluator] = None


def get_quality_evaluator() -> AgentQualityEvaluator:
    """Get global quality evaluator instance"""
    global _quality_evaluator
    if _quality_evaluator is None:
        _quality_evaluator = AgentQualityEvaluator()
    return _quality_evaluator


def evaluate_agent_quality(agent_id: str = None, **kwargs) -> Dict[str, Any]:
    """
    REAL function for agent quality evaluation
    
    Args:
        agent_id: Agent identifier (required)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with real quality metrics
    """
    if agent_id is None:
        raise ValueError("agent_id is required for quality evaluation")
    
    evaluator = get_quality_evaluator()
    return evaluator.evaluate_agent_quality(agent_id, **kwargs)
