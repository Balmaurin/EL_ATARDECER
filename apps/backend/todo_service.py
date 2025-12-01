import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import enum

class TodoService:
    def __init__(self):
        self.data_dir = Path("data/todos")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.todos_file = self.data_dir / "todos.json"
        self.projects_file = self.data_dir / "projects.json"
        self.todos = []
        self.projects = []
        self._load()

    def _load(self):
        if self.todos_file.exists():
            try:
                with open(self.todos_file, 'r') as f:
                    self.todos = json.load(f)
            except:
                self.todos = []
        else:
            self.todos = []

        if self.projects_file.exists():
            try:
                with open(self.projects_file, 'r') as f:
                    self.projects = json.load(f)
            except:
                self.projects = []
        else:
            self.projects = []

    def _save(self):
        with open(self.todos_file, 'w') as f:
            json.dump(self.todos, f, indent=2)
        with open(self.projects_file, 'w') as f:
            json.dump(self.projects, f, indent=2)

    def get_todos(self, status: str = None, priority: str = None, category: str = None, project_id: str = None) -> List[Dict]:
        filtered = self.todos
        if status:
            filtered = [t for t in filtered if t.get('status') == status]
        if priority:
            filtered = [t for t in filtered if t.get('priority') == priority]
        if category:
            filtered = [t for t in filtered if t.get('category') == category]
        if project_id:
            filtered = [t for t in filtered if t.get('project_id') == project_id]
        return filtered

    def get_todo(self, todo_id: str) -> Optional[Dict]:
        for todo in self.todos:
            if todo['id'] == todo_id:
                return todo
        return None

    def create_todo(self, todo_data: Dict) -> Dict:
        # Ensure ID
        if 'id' not in todo_data:
            todo_data['id'] = f"todo-{int(datetime.now().timestamp())}"
        
        # Set defaults
        todo_data.setdefault('created_at', datetime.now().isoformat())
        todo_data.setdefault('updated_at', datetime.now().isoformat())
        todo_data.setdefault('status', 'pending')
        
        self.todos.append(todo_data)
        self._save()
        return todo_data

    def update_todo(self, todo_id: str, updates: Dict) -> Optional[Dict]:
        for i, todo in enumerate(self.todos):
            if todo['id'] == todo_id:
                # Remove None values from updates
                clean_updates = {k: v for k, v in updates.items() if v is not None}
                self.todos[i].update(clean_updates)
                self.todos[i]['updated_at'] = datetime.now().isoformat()
                
                # Handle completion
                if clean_updates.get('status') == 'completed' and not self.todos[i].get('completed_at'):
                    self.todos[i]['completed_at'] = datetime.now().isoformat()
                
                self._save()
                return self.todos[i]
        return None

    def delete_todo(self, todo_id: str) -> bool:
        initial_len = len(self.todos)
        self.todos = [t for t in self.todos if t['id'] != todo_id]
        if len(self.todos) < initial_len:
            self._save()
            return True
        return False

    def get_projects(self, status: str = None) -> List[Dict]:
        filtered = self.projects
        if status:
            filtered = [p for p in filtered if p.get('status') == status]
        return filtered

    def create_project(self, project_data: Dict) -> Dict:
        if 'id' not in project_data:
            project_data['id'] = f"project-{int(datetime.now().timestamp())}"
        
        project_data.setdefault('created_at', datetime.now().isoformat())
        project_data.setdefault('updated_at', datetime.now().isoformat())
        project_data.setdefault('status', 'in_progress')
        project_data.setdefault('progress_percentage', 0.0)
        
        self.projects.append(project_data)
        self._save()
        return project_data

    def update_project(self, project_id: str, updates: Dict) -> Optional[Dict]:
        for i, project in enumerate(self.projects):
            if project['id'] == project_id:
                clean_updates = {k: v for k, v in updates.items() if v is not None}
                self.projects[i].update(clean_updates)
                self.projects[i]['updated_at'] = datetime.now().isoformat()
                self._save()
                return self.projects[i]
        return None

    def get_stats(self) -> Dict:
        total = len(self.todos)
        completed = len([t for t in self.todos if t.get('status') == 'completed'])
        pending = len([t for t in self.todos if t.get('status') == 'pending'])
        in_progress = len([t for t in self.todos if t.get('status') == 'in_progress'])
        
        # Calculate completion rate
        rate = (completed / total * 100) if total > 0 else 0.0
        
        # Group by priority
        by_priority = {}
        for t in self.todos:
            p = t.get('priority', 'medium')
            by_priority[p] = by_priority.get(p, 0) + 1
            
        # Group by category
        by_category = {}
        for t in self.todos:
            c = t.get('category', 'other')
            by_category[c] = by_category.get(c, 0) + 1

        # Calculate overdue todos
        overdue_count = 0
        now = datetime.now()
        for t in self.todos:
            if t.get('status') in ['pending', 'in_progress']:
                due_date_str = t.get('due_date')
                if due_date_str:
                    try:
                        due_date = datetime.fromisoformat(due_date_str.replace('Z', '+00:00'))
                        if due_date.tzinfo is None:
                            due_date = now.replace(tzinfo=None)
                        if due_date < now:
                            overdue_count += 1
                    except (ValueError, AttributeError):
                        pass
        
        # Calculate average completion time
        avg_completion_time = 0.0
        completed_with_times = []
        for t in self.todos:
            if t.get('status') == 'completed':
                created_str = t.get('created_at')
                completed_str = t.get('completed_at')
                if created_str and completed_str:
                    try:
                        created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                        completed = datetime.fromisoformat(completed_str.replace('Z', '+00:00'))
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=None)
                        if completed.tzinfo is None:
                            completed = completed.replace(tzinfo=None)
                        time_diff = (completed - created).total_seconds() / 3600  # hours
                        if time_diff > 0:
                            completed_with_times.append(time_diff)
                    except (ValueError, AttributeError):
                        pass
        
        if completed_with_times:
            avg_completion_time = sum(completed_with_times) / len(completed_with_times)

        return {
            "total_todos": total,
            "completed_todos": completed,
            "pending_todos": pending,
            "in_progress_todos": in_progress,
            "overdue_todos": overdue_count,
            "completion_rate": rate,
            "average_completion_time": avg_completion_time,
            "todos_by_priority": by_priority,
            "todos_by_category": by_category
        }
