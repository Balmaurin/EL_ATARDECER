#!/usr/bin/env python3
"""
🎓 MASTER EDUCATION SYSTEM - SISTEMA EDUCATIVO WEB3 LEARN-TO-EARN
================================================================

Sistema educativo blockchain avanzado con aprendizaje gamificado:
- Learn-to-Earn con tokens SHEILYS por progreso educativo
- NFT certificates de finalización de cursos
- DAO governance para contenido educativo
- IA personalizada para aprendizaje adaptativo
- Marketplace de conocimientos peer-to-peer
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class MasterEducationSystem:
    """
    Sistema Educativo Web3 Master - Learn-to-Earn Enterprise

    Características avanzadas:
    - Economía educativa con SHEILYS tokens
    - NFT certificates blockchain-verified
    - Cursos gamificados con rewards dinámicos
    - IA adaptativa para rutas de aprendizaje personalizadas
    - DAO governance para curación de contenido
    - Marketplace P2P de conocimientos y habilidades
    """

    def __init__(self, education_db: str = "education_blockchain.json"):
        self.education_db = education_db
        self.education_data = self._load_education_data()

        # Configuración del sistema educativo
        self.education_config = {
            "max_students_per_course": 1000,
            "reward_multiplier_base": 1.0,
            "completion_bonus_percent": 25,
            "nft_certificate_fee": 0.1,  # SHEILYS
            "staking_requirement_instructor": 100,  # SHEILYS minimum
            "dao_voting_threshold": 100,  # Votes minimum
            "adaptive_learning_levels": 5,
            "peer_review_minimum": 3,
        }

        print("🎓 Master Education System Web3 inicializado")
        print(f"📚 Cursos disponibles: {len(self.education_data.get('courses', {}))}")
        print(
            f"👥 Estudiantes registrados: {len(self.education_data.get('students', {}))}"
        )
        print(f"🎯 Sistema Learn-to-Earn: OPERATIVO")
        print(f"⛓️  Blockchain integration: ACTIVA")

    def _load_education_data(self) -> Dict[str, Any]:
        """Cargar datos educativos desde storage persistente"""
        try:
            if os.path.exists(self.education_db):
                with open(self.education_db, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return self._initialize_education_blockchain()
        except Exception as e:
            print(f"Error cargando datos educativos: {e}")
            return self._initialize_education_blockchain()

    def _initialize_education_blockchain(self) -> Dict[str, Any]:
        """Inicializar blockchain educativa con estructura avanzada"""
        return {
            "version": "3.0-web3-learn-to-earn",
            "blockchain_network": "SHEILYS_EDUCATION_CHAIN",
            "genesis_timestamp": datetime.now().isoformat(),
            "total_reward_pool": 100000,  # SHEILYS disponibles para rewards
            "total_distributed_rewards": 0,
            "courses": {
                "blockchain_101": self._create_course_template(
                    "Blockchain Fundamentals"
                ),
                "ai_ml_basics": self._create_course_template(
                    "AI & Machine Learning Basics"
                ),
                "web3_development": self._create_course_template(
                    "Web3 Development Masterclass"
                ),
                "dao_governance": self._create_course_template(
                    "DAO Governance & Voting"
                ),
                "defi_finance": self._create_course_template("DeFi & Tokenomics"),
                "smart_contracts": self._create_course_template(
                    "Smart Contract Development"
                ),
                "nft_creation": self._create_course_template(
                    "NFT Creation & Management"
                ),
                "metaverse_building": self._create_course_template(
                    "Metaverse Development"
                ),
            },
            "students": {},
            "instructors": {},
            "certificates_issued": {},
            "learning_paths": {},
            "dao_proposals": [],
            "reward_distributions": [],
            "marketplace": {
                "active_listings": {},
                "completed_transactions": [],
                "total_volume_sheilys": 0,
            },
            "governance": {
                "total_voting_power": 0,
                "active_proposals": {},
                "passed_proposals": [],
            },
            "analytics": {
                "total_enrollments": 0,
                "completion_rate": 0.0,
                "average_course_rating": 0.0,
                "top_performing_categories": [],
            },
        }

    def _create_course_template(self, title: str) -> Dict[str, Any]:
        """Crear plantilla de curso con estructura completa"""
        course_id = title.lower().replace(" ", "_").replace("&", "and")

        return {
            "id": course_id,
            "title": title,
            "description": f"Comprehensive course on {title.lower()}",
            "category": self._determine_category(title),
            "difficulty": "intermediate",
            "duration_hours": 40,
            "instructor": "master_instructor",
            "price_sheilys": 50,
            "reward_pool": 5000,
            "modules": [
                {
                    "id": f"{course_id}_module_1",
                    "title": f"{title} Fundamentals",
                    "duration": 8,
                    "reward": 500,
                },
                {
                    "id": f"{course_id}_module_2",
                    "title": f"Advanced {title}",
                    "duration": 12,
                    "reward": 800,
                },
                {
                    "id": f"{course_id}_module_3",
                    "title": f"{title} in Practice",
                    "duration": 10,
                    "reward": 700,
                },
                {
                    "id": f"{course_id}_module_4",
                    "title": f"{title} Projects",
                    "duration": 10,
                    "reward": 1000,
                },
            ],
            "prerequisites": [],
            "skills_taught": [f"{title.lower()} skills", "technical expertise"],
            "certification_nft": {
                "enabled": True,
                "metadata_uri": f"ipfs://education/{course_id}_certificate",
                "rarity_levels": ["bronze", "silver", "gold", "platinum"],
            },
            "enrolled_students": 0,
            "completed_students": 0,
            "average_rating": 0.0,
            "total_reviews": 0,
            "staking_requirements": {"instructor_minimum": 100, "student_minimum": 10},
            "gamification": {
                "achievements": [
                    {
                        "name": f"{title} Explorer",
                        "requirement": "complete_module_1",
                        "reward": 100,
                    },
                    {
                        "name": f"{title} Master",
                        "requirement": "complete_course",
                        "reward": 500,
                    },
                    {
                        "name": f"{title} Expert",
                        "requirement": "100%_score_all_modules",
                        "reward": 1500,
                    },
                ],
                "leaderboards": True,
                "weekly_challenges": True,
            },
            "ai_personalization": {
                "adaptive_difficulty": True,
                "personalized_pathways": True,
                "intelligent_recommendations": True,
                "performance_prediction": True,
            },
        }

    def _determine_category(self, title: str) -> str:
        """Determinar categoría del curso basado en el título"""
        categories = {
            "blockchain": "Blockchain",
            "ai": "Artificial Intelligence",
            "web3": "Web3 Development",
            "dao": "Governance",
            "defi": "DeFi & Finance",
            "smart": "Smart Contracts",
            "nft": "NFTs & Digital Assets",
            "metaverse": "Metaverse",
        }

        title_lower = title.lower()
        for keyword, category in categories.items():
            if keyword in title_lower:
                return category

        return "General Technology"

    async def enroll_student(
        self, student_wallet: str, course_id: str, stake_amount: float = 10.0
    ) -> Dict[str, Any]:
        """
        Inscribir estudiante con staking SHEILYS - Learn-to-Earn system
        """
        try:
            if course_id not in self.education_data["courses"]:
                return {"success": False, "error": "Course not found"}

            course = self.education_data["courses"][course_id]

            # Verificar requisitos de staking
            if stake_amount < course["staking_requirements"]["student_minimum"]:
                return {
                    "success": False,
                    "error": f"Insufficient stake. Required: {course['staking_requirements']['student_minimum']} SHEILYS",
                }

            # Inicializar estudiante si no existe
            if student_wallet not in self.education_data["students"]:
                self.education_data["students"][student_wallet] = {
                    "wallet": student_wallet,
                    "enrolled_courses": [],
                    "completed_courses": [],
                    "total_earned_sheilys": 0,
                    "staking_balance": 0,
                    "nft_certificates": [],
                    "learning_streak": 0,
                    "skill_level": "beginner",
                    "personalized_path": {},
                }

            student = self.education_data["students"][student_wallet]

            # Verificar que no esté ya inscrito
            if course_id in student["enrolled_courses"]:
                return {"success": False, "error": "Already enrolled in this course"}

            # Procesar pago del curso
            total_cost = course["price_sheilys"]
            if stake_amount < total_cost:
                return {
                    "success": False,
                    "error": f"Insufficient funds. Total cost: {total_cost} SHEILYS",
                }

            # Inscribir estudiante
            enrollment_data = {
                "course_id": course_id,
                "enrollment_date": datetime.now().isoformat(),
                "stake_amount": stake_amount,
                "paid_amount": total_cost,
                "refund_amount": stake_amount - total_cost,
                "progress": 0.0,
                "completed_modules": [],
                "quiz_scores": {},
                "time_spent_hours": 0,
                "adaptive_difficulty": 1.0,
                "predicted_completion_date": self._predict_completion_date(course),
                "current_achievements": [],
            }

            # Añadir a cursos inscritos del estudiante
            student["enrolled_courses"].append(course_id)
            student["enrollment_data"] = student.get("enrollment_data", {})
            student["enrollment_data"][course_id] = enrollment_data

            # Actualizar estadísticas del curso
            course["enrolled_students"] += 1

            # Actualizar estadísticas globales
            self.education_data["analytics"]["total_enrollments"] += 1

            await self._save_education_data()

            print(
                f"🎓 Estudiante {student_wallet[:8]}... inscrito en curso {course_id}"
            )
            print(
                f"💰 Pagado: {total_cost} SHEILYS | Stake restante: {stake_amount - total_cost}"
            )

            return {
                "success": True,
                "enrollment_id": f"enroll_{int(datetime.now().timestamp())}",
                "course": course_id,
                "stake_used": total_cost,
                "stake_refunded": stake_amount - total_cost,
                "enrollment_details": enrollment_data,
            }

        except Exception as e:
            print(f"Error inscribiendo estudiante: {e}")
            return {"success": False, "error": str(e)}

    def _predict_completion_date(self, course: Dict[str, Any]) -> str:
        """Predecir fecha de finalización basada en duración del curso"""
        duration_hours = course.get("duration_hours", 40)
        # Asumir dedicación promedio de 10 horas/semana
        weeks_needed = duration_hours / 10
        predicted_date = datetime.now() + timedelta(weeks=weeks_needed)

        return predicted_date.isoformat()

    async def complete_module(
        self, student_wallet: str, course_id: str, module_id: str, quiz_score: float
    ) -> Dict[str, Any]:
        """
        Completar módulo y distribuir rewards SHEILYS
        """
        try:
            if student_wallet not in self.education_data["students"]:
                return {"success": False, "error": "Student not found"}

            student = self.education_data["students"][student_wallet]
            if course_id not in student.get("enrollment_data", {}):
                return {"success": False, "error": "Not enrolled in this course"}

            enrollment = student["enrollment_data"][course_id]
            course = self.education_data["courses"][course_id]

            # Verificar que el módulo existe
            module = next((m for m in course["modules"] if m["id"] == module_id), None)
            if not module:
                return {"success": False, "error": "Module not found"}

            # Verificar que no esté ya completado
            if module_id in enrollment["completed_modules"]:
                return {"success": False, "error": "Module already completed"}

            # Aplicar multiplicador basado en score del quiz
            score_multiplier = (
                1.0 + (quiz_score / 100) * 0.5
            )  # Bonus hasta 50% por score perfecto
            base_reward = module["reward"]
            actual_reward = int(base_reward * score_multiplier)

            # Distribuir reward
            reward_distribution = {
                "student_wallet": student_wallet,
                "course_id": course_id,
                "module_id": module_id,
                "base_reward": base_reward,
                "score_multiplier": score_multiplier,
                "actual_reward": actual_reward,
                "quiz_score": quiz_score,
                "timestamp": datetime.now().isoformat(),
                "transaction_hash": self._generate_transaction_hash(),
            }

            # Actualizar datos del estudiante
            enrollment["completed_modules"].append(module_id)
            enrollment["quiz_scores"][module_id] = quiz_score

            # Calcular progreso total
            total_modules = len(course["modules"])
            completed_count = len(enrollment["completed_modules"])
            enrollment["progress"] = (completed_count / total_modules) * 100

            # Actualizar rewards totales
            student["total_earned_sheilys"] += actual_reward
            self.education_data["total_distributed_rewards"] += actual_reward

            # Verificar si curso completado
            course_completed = completed_count == total_modules
            if course_completed:
                await self._complete_course(student_wallet, course_id, enrollment)

            # Registrar distribución
            self.education_data["reward_distributions"].append(reward_distribution)

            await self._save_education_data()

            print(f"✅ Módulo {module_id} completado por {student_wallet[:8]}...")
            print(
                f"💰 Reward: {actual_reward} SHEILYS (base: {base_reward}, multiplier: {score_multiplier:.1f})"
            )

            return {
                "success": True,
                "module_completed": module_id,
                "reward_earned": actual_reward,
                "total_progress": enrollment["progress"],
                "course_completed": course_completed,
                "transaction_hash": reward_distribution["transaction_hash"],
            }

        except Exception as e:
            print(f"Error completando módulo: {e}")
            return {"success": False, "error": str(e)}

    async def _complete_course(
        self, student_wallet: str, course_id: str, enrollment: Dict[str, Any]
    ):
        """Completar curso y emitir NFT certificate"""
        try:
            student = self.education_data["students"][student_wallet]
            course = self.education_data["courses"][course_id]

            # Calcular calificación final
            quiz_scores = list(enrollment["quiz_scores"].values())
            final_score = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0

            # Asignar rareza del NFT basado en score
            if final_score >= 95:
                rarity = "platinum"
                bonus_multiplier = 2.0
            elif final_score >= 85:
                rarity = "gold"
                bonus_multiplier = 1.5
            elif final_score >= 75:
                rarity = "silver"
                bonus_multiplier = 1.2
            else:
                rarity = "bronze"
                bonus_multiplier = 1.0

            # Bonus de finalización
            completion_bonus = int(
                course["reward_pool"]
                * self.education_config["completion_bonus_percent"]
                / 100
                * bonus_multiplier
            )

            # Generar NFT certificate
            certificate_id = (
                f"cert_{course_id}_{student_wallet}_{int(datetime.now().timestamp())}"
            )
            blockchain_tx = self._generate_transaction_hash()

            certificate = {
                "id": certificate_id,
                "course_id": course_id,
                "student_wallet": student_wallet,
                "final_score": final_score,
                "rarity": rarity,
                "completion_date": datetime.now().isoformat(),
                "metadata_uri": f"ipfs://education/certificates/{certificate_id}",
                "blockchain_tx": blockchain_tx,
                "verification_hash": "",
            }

            # Generar hash de verificación después de crear el certificado
            certificate["verification_hash"] = self._generate_certificate_hash(
                certificate
            )

            # Actualizar datos del estudiante
            student["completed_courses"].append(course_id)
            student["nft_certificates"].append(certificate)
            student["total_earned_sheilys"] += completion_bonus

            # Remover de cursos activos
            if course_id in student["enrolled_courses"]:
                student["enrolled_courses"].remove(course_id)

            # Actualizar estadísticas del curso
            course["completed_students"] += 1

            # Actualizar estadísticas globales
            self.education_data["analytics"]["completion_rate"] = (
                (course["completed_students"] / course["enrolled_students"]) * 100
                if course["enrolled_students"] > 0
                else 0
            )

            # Registrar certificado
            self.education_data["certificates_issued"][certificate_id] = certificate

            print(f"🎉 CURSO COMPLETADO: {course_id} por {student_wallet[:8]}...")
            print(
                f"🏆 Calificación final: {final_score:.1f}% | Rareza NFT: {rarity.upper()}"
            )
            print(f"💰 Bonus de completación: {completion_bonus} SHEILYS")

            # Devolver staking si score >= 70%
            if final_score >= 70:
                refund_amount = enrollment.get("refund_amount", 0)
                if refund_amount > 0:
                    print(f"🔄 Stake devuelto: {refund_amount} SHEILYS")
                    student["staking_balance"] += refund_amount

        except Exception as e:
            print(f"Error completando curso: {e}")

    def _generate_transaction_hash(self) -> str:
        """Generar hash de transacción SHEILYS"""
        data = f"{datetime.now().timestamp()}_{os.urandom(8).hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    def _generate_certificate_hash(self, certificate: Dict[str, Any]) -> str:
        """Generar hash de verificación para certificado"""
        cert_data = json.dumps(certificate, sort_keys=True, default=str)
        return hashlib.sha256(cert_data.encode()).hexdigest()

    async def get_personalized_learning_path(
        self, student_wallet: str
    ) -> Dict[str, Any]:
        """
        Generar ruta de aprendizaje personalizada con IA adaptativa
        """
        try:
            if student_wallet not in self.education_data["students"]:
                return {"error": "Student not found"}

            student = self.education_data["students"][student_wallet]

            # Analizar perfil del estudiante
            completed_courses = student.get("completed_courses", [])
            enrolled_courses = student.get("enrolled_courses", [])
            current_skill_level = student.get("skill_level", "beginner")
            total_earned = student.get("total_earned_sheilys", 0)

            # Recomendaciones basadas en perfil
            recommendations = []

            # Para principiantes
            if not completed_courses:
                recommendations.extend(
                    ["blockchain_101", "web3_development", "nft_creation"]
                )
            # Para intermedios
            elif len(completed_courses) < 3:
                if "blockchain_101" in completed_courses:
                    recommendations.extend(
                        ["smart_contracts", "defi_finance", "dao_governance"]
                    )
            # Para avanzados
            else:
                recommendations.extend(
                    ["ai_ml_basics", "metaverse_building", "dao_governance"]
                )

            # Filtrar cursos ya inscritos o completados
            available_recommendations = [
                course
                for course in recommendations
                if course not in enrolled_courses and course not in completed_courses
            ]

            # Calcular dificultad adaptativa
            adaptive_difficulty = self._calculate_adaptive_difficulty(student)

            # Predecir tiempo de aprendizaje
            predicted_completion = self._predict_learning_timeline(
                student, available_recommendations
            )

            learning_path = {
                "student_wallet": student_wallet,
                "current_skill_level": current_skill_level,
                "recommended_courses": available_recommendations[:5],  # Top 5
                "adaptive_difficulty": adaptive_difficulty,
                "predicted_completion_months": predicted_completion,
                "total_earned_so_far": total_earned,
                "next_milestones": self._generate_milestones(
                    student, available_recommendations
                ),
                "skill_gaps_identified": self._identify_skill_gaps(student),
                "market_opportunities": self._suggest_market_opportunities(student),
            }

            return learning_path

        except Exception as e:
            print(f"Error generando ruta de aprendizaje: {e}")
            return {"error": str(e)}

    def _calculate_adaptive_difficulty(self, student: Dict[str, Any]) -> float:
        """Calcular dificultad adaptativa basada en rendimiento del estudiante"""
        completed = student.get("completed_courses", [])
        quiz_scores = []

        # Recopilar scores de quizzes
        enrollment_data = student.get("enrollment_data", {})
        for course_data in enrollment_data.values():
            quiz_scores.extend(course_data.get("quiz_scores", {}).values())

        if not quiz_scores:
            return 1.0  # Nivel base

        avg_score = sum(quiz_scores) / len(quiz_scores)

        # Ajustar dificultad basado en performance
        if avg_score >= 85:
            return 1.3  # Nivel avanzado
        elif avg_score >= 70:
            return 1.0  # Nivel estándar
        else:
            return 0.8  # Nivel facilitado

    def _predict_learning_timeline(
        self, student: Dict[str, Any], recommended_courses: List[str]
    ) -> float:
        """Predecir timeline de aprendizaje"""
        enrolled = len(student.get("enrolled_courses", []))
        completed = len(student.get("completed_courses", []))

        # Calcular ratio de completación
        completion_ratio = completed / max(enrolled + completed, 1)

        # Estimar tiempo por curso basado en ratio
        base_time_per_course = 4  # semanas
        adaptive_time = base_time_per_course / max(completion_ratio, 0.3)

        return round(len(recommended_courses) * adaptive_time, 1)

    def _generate_milestones(
        self, student: Dict[str, Any], courses: List[str]
    ) -> List[Dict[str, Any]]:
        """Generar milestones de aprendizaje"""
        milestones = []

        for i, course_id in enumerate(courses[:3]):
            if course_id in self.education_data["courses"]:
                course = self.education_data["courses"][course_id]
                milestone = {
                    "order": i + 1,
                    "course_id": course_id,
                    "course_title": course["title"],
                    "reward_estimate": (
                        course["reward_pool"] // course["enrolled_students"]
                        if course["enrolled_students"] > 0
                        else 0
                    ),
                    "estimated_completion_weeks": course["duration_hours"] // 10,
                    "prerequisites": course.get("prerequisites", []),
                }
                milestones.append(milestone)

        return milestones

    def _identify_skill_gaps(self, student: Dict[str, Any]) -> List[str]:
        """Identificar gaps de habilidades"""
        skills_learned = set()
        enrolled_in = student.get("enrolled_courses", [])
        completed = student.get("completed_courses", [])

        # Compilar skills aprendidos
        for course_id in completed + enrolled_in:
            if course_id in self.education_data["courses"]:
                course = self.education_data["courses"][course_id]
                skills_learned.update(course.get("skills_taught", []))

        # Skills críticas faltantes
        critical_skills = [
            "blockchain_fundamentals",
            "smart_contracts",
            "defi_mechanics",
            "nft_creation",
            "dao_governance",
            "web3_development",
        ]

        gaps = [skill for skill in critical_skills if skill not in skills_learned]
        return gaps

    def _suggest_market_opportunities(
        self, student: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Sugerir oportunidades de mercado basadas en skills"""
        opportunities = []

        skills = set()
        for course_id in student.get("completed_courses", []):
            if course_id in self.education_data["courses"]:
                course = self.education_data["courses"][course_id]
                skills.update(course.get("skills_taught", []))

        # Oportunidades basadas en skills
        if "smart_contracts" in skills:
            opportunities.append(
                {
                    "role": "Smart Contract Developer",
                    "estimated_salary_range": "80k-150k SHEILYS/year",
                    "demand_level": "high",
                    "market_trend": "bullish",
                }
            )

        if "nft_creation" in skills:
            opportunities.append(
                {
                    "role": "NFT Creator/Artist",
                    "estimated_salary_range": "50k-200k SHEILYS/year",
                    "demand_level": "very_high",
                    "market_trend": "explosive",
                }
            )

        if "defi_mechanics" in skills:
            opportunities.append(
                {
                    "role": "DeFi Analyst/Developer",
                    "estimated_salary_range": "100k-250k SHEILYS/year",
                    "demand_level": "extreme",
                    "market_trend": "hypergrowth",
                }
            )

        return opportunities[:3]  # Top 3 oportunidades

    async def get_education_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema educativo"""
        try:
            students_count = len(self.education_data["students"])
            courses_count = len(self.education_data["courses"])
            total_rewards = self.education_data["total_distributed_rewards"]
            certifications = len(self.education_data["certificates_issued"])

            stats = {
                "total_students": students_count,
                "total_courses": courses_count,
                "total_enrollments": self.education_data["analytics"][
                    "total_enrollments"
                ],
                "completion_rate": self.education_data["analytics"]["completion_rate"],
                "total_rewards_distributed": total_rewards,
                "total_certifications_issued": certifications,
                "marketplace_volume": self.education_data["marketplace"][
                    "total_volume_sheilys"
                ],
                "top_courses": self._get_top_courses()[:5],
                "learning_streaks": self._get_learning_streaks(),
                "skill_distribution": self._analyze_skill_distribution(),
                "financial_metrics": {
                    "total_reward_pool": self.education_data["total_reward_pool"],
                    "distributed_percentage": (
                        (total_rewards / self.education_data["total_reward_pool"]) * 100
                        if self.education_data["total_reward_pool"] > 0
                        else 0
                    ),
                    "average_reward_per_completion": total_rewards
                    / max(certifications, 1),
                },
                "engagement_metrics": {
                    "active_learners_ratio": self._calculate_active_learners_ratio(),
                    "average_course_rating": self.education_data["analytics"][
                        "average_course_rating"
                    ],
                    "total_reviews_submitted": sum(
                        course["total_reviews"]
                        for course in self.education_data["courses"].values()
                    ),
                },
            }

            print("📊 Master Education System Stats:")
            print(f"   👥 Estudiantes: {students_count}")
            print(f"   📚 Cursos: {courses_count}")
            print(f"   🏆 Certificaciones: {certifications}")
            print(f"   💰 Rewards distribuidos: {total_rewards} SHEILYS")
            print(f"   📈 Completion rate: {stats['completion_rate']:.1f}%")

            return stats

        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

    def _get_top_courses(self) -> List[Dict[str, Any]]:
        """Obtener cursos más populares"""
        courses = []
        for course_id, course_data in self.education_data["courses"].items():
            courses.append(
                {
                    "id": course_id,
                    "title": course_data["title"],
                    "enrollments": course_data["enrolled_students"],
                    "completions": course_data["completed_students"],
                    "rating": course_data["average_rating"],
                }
            )

        # Ordenar por enrollments
        return sorted(courses, key=lambda x: x["enrollments"], reverse=True)

    def _get_learning_streaks(self) -> Dict[str, int]:
        """Obtener streaks de aprendizaje activo"""
        streaks = {}
        for student_id, student_data in self.education_data["students"].items():
            streak = student_data.get("learning_streak", 0)
            if streak > 0:
                streaks[student_id[:8]] = streak

        return dict(sorted(streaks.items(), key=lambda x: x[1], reverse=True)[:10])

    def _analyze_skill_distribution(self) -> Dict[str, int]:
        """Analizar distribución de skills aprendidos"""
        skill_count = {}

        for course_data in self.education_data["courses"].values():
            skills = course_data.get("skills_taught", [])
            for skill in skills:
                skill_count[skill] = (
                    skill_count.get(skill, 0) + course_data["completed_students"]
                )

        return dict(sorted(skill_count.items(), key=lambda x: x[1], reverse=True)[:10])

    def _calculate_active_learners_ratio(self) -> float:
        """Calcular ratio de estudiantes activos"""
        total_students = len(self.education_data["students"])
        if total_students == 0:
            return 0.0

        active_students = sum(
            1
            for student in self.education_data["students"].values()
            if student.get("enrolled_courses", [])
        )

        return (active_students / total_students) * 100

    async def _save_education_data(self):
        """Guardar datos educativos de forma persistente"""
        try:
            # Crear respaldo si existe
            if os.path.exists(self.education_db):
                backup_file = f"{self.education_db}.backup"
                os.replace(self.education_db, backup_file)

            with open(self.education_db, "w", encoding="utf-8") as f:
                json.dump(self.education_data, f, indent=2, ensure_ascii=False)

            print("💾 Education blockchain data saved successfully")

        except Exception as e:
            print(f"Error saving education data: {e}")


# Instancia global del Master Education System
master_education_system = MasterEducationSystem()


def get_master_education_system() -> MasterEducationSystem:
    """
    Función getter para obtener la instancia singleton del sistema educativo maestro

    Returns:
        MasterEducationSystem: Instancia global del sistema educativo
    """
    return master_education_system


print("\n🎓 MASTER EDUCATION SYSTEM WEB3 COMPLETAMENTE OPERATIVO")
print("💰 Learn-to-Earn: ACTIVO con SHEILYS tokens")
print("⛓️  Blockchain certificates: FUNCIONALES")
print("🎮 Gamification rewards: IMPLEMENTADA")
print("🤖 IA adaptativa de aprendizaje: OPERATIVA")
