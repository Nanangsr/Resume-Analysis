from typing import List, Dict, Optional
import logging
import re
from utils.resume_standardizer import ResumeStandardizer

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScorer:
    def __init__(self, domain: str = "general", criteria: Optional[Dict[str, int]] = None):
        """
        Initialize rule-based resume scorer with domain flexibility
        
        Args:
            domain: Target domain (it, hr, finance, marketing, general, etc.)
            criteria: Dictionary mapping criteria names to their weights
        """
        self.domain = domain.lower()
        self.standardizer = ResumeStandardizer(domain=self.domain)
        
        # Gunakan kriteria domain-specific dari standardizer jika tidak ada kriteria yang diberikan
        self.criteria = criteria if criteria else self.standardizer.get_domain_specific_criteria()
        self.max_score = sum(self.criteria.values())
        
        # Dapatkan pemetaan keterampilan domain yang sesuai
        self.domain_skills = self.standardizer.domain_skills_mapping.get(
            self.domain.upper(), 
            self.standardizer.domain_skills_mapping["General"]
        )
        
        # Mapping untuk interpretasi skor
        self.scoring_guide = {
            1: "Tidak memenuhi",
            2: "Tidak memenuhi", 
            3: "Tidak memenuhi",
            4: "Memenuhi sebagian",
            5: "Memenuhi sebagian",
            6: "Memenuhi sebagian",
            7: "Memenuhi dengan baik",
            8: "Memenuhi dengan baik",
            9: "Melebihi ekspektasi",
            10: "Melebihi ekspektasi"
        }
        
        # Level expectations untuk konteks
        self.level_expectations = {
            "entry": {"min_score": 20, "max_score": 50},
            "mid": {"min_score": 40, "max_score": 70},
            "senior": {"min_score": 60, "max_score": 90},
            "expert": {"min_score": 80, "max_score": 100}
        }
    
    def extract_features_from_resume(self, resume_text: str, jd_text: Optional[str] = None) -> Dict:
        """
        Extract features from resume text with domain awareness
        
        Args:
            resume_text: Raw resume text
            jd_text: Job description text for skill matching (optional)
            
        Returns:
            Dictionary with extracted features
        """
        try:
            # Standardize resume first
            standardized_resume = self.standardizer.standardize_resume(resume_text)
            
            # Extract skills from standardized resume
            skills_match = re.search(r'SKILLS:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume, re.DOTALL)
            if skills_match:
                skills_text = skills_match.group(1).strip()
                candidate_skills = [skill.strip().lower() for skill in skills_text.split(',')]
            else:
                candidate_skills = []
            
            # Calculate skill match score if job description is provided
            skill_match_score = 0
            if jd_text:
                # Extract skills from job description menggunakan domain_skills yang sesuai
                jd_skills = []
                for skill in self.domain_skills:  # Menggunakan keterampilan domain yang telah disesuaikan
                    if skill.lower() in jd_text.lower():
                        jd_skills.append(skill.lower())
                
                if jd_skills:
                    matched_skills = set(jd_skills) & set(candidate_skills)
                    skill_match_score = len(matched_skills) / len(jd_skills) if jd_skills else 0
                else:
                    # Fallback jika tidak ada keterampilan domain di JD
                    skill_match_score = len(candidate_skills) / 10 if candidate_skills else 0
            else:
                skill_match_score = len(candidate_skills) / 10 if candidate_skills else 0
            
            # Extract experience years
            experience_match = re.search(r'EXPERIENCE_YEARS:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume)
            if experience_match:
                experience_value = experience_match.group(1).strip()
                if experience_value.lower() == 'not specified':
                    experience_years = 0  # Nilai default jika 'Not specified'
                else:
                    try:
                        experience_years = int(experience_value)
                    except ValueError:
                        experience_years = 0  # Nilai default jika konversi gagal
            else:
                experience_years = 0  # Nilai default jika tidak ditemukan
            
            # Extract education level
            education_match = re.search(r'EDUCATION:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume, re.DOTALL)
            if education_match:
                education_text = education_match.group(1).strip().lower()
                if "phd" in education_text or "doctorate" in education_text:
                    education_level = 4
                elif "master" in education_text or "mba" in education_text:
                    education_level = 3
                elif "bachelor" in education_text or "bsc" in education_text:
                    education_level = 2
                else:
                    education_level = 1
            else:
                education_level = 0
            
            # Extract certifications
            cert_match = re.search(r'CERTIFICATIONS:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume, re.DOTALL)
            if cert_match:
                cert_text = cert_match.group(1).strip()
                certifications_count = len(cert_text.split(',')) if cert_text != "None" else 0
            else:
                certifications_count = 0
            
            # Extract projects count
            projects_match = re.search(r'PROJECTS_COUNT:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume)
            if projects_match:
                projects_count = int(projects_match.group(1).strip())
            else:
                projects_count = 0
            
            # Extract salary expectation
            salary_match = re.search(r'SALARY_EXPECTATION:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume)
            if salary_match:
                salary_text = salary_match.group(1).strip()
                if salary_text != "Not specified":
                    salary_expectation = float(re.sub(r'[^\d.]', '', salary_text))
                else:
                    salary_expectation = 0.0
            else:
                salary_expectation = 0.0
            
            # Extract domain expertise
            expertise_match = re.search(r'DOMAIN_EXPERTISE:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume)
            if expertise_match:
                expertise_text = expertise_match.group(1).strip().lower()
                expertise_score = 0.8 if "advanced" in expertise_text or "expert" in expertise_text else \
                                0.5 if "intermediate" in expertise_text else \
                                0.2 if "entry" in expertise_text else 0.0
            else:
                expertise_score = 0.0
            
            # Detect job role/level
            job_role_match = re.search(r'JOB_ROLE:(.+?)(?=\n[A-Z_]+:|$)', standardized_resume)
            if job_role_match:
                job_role = job_role_match.group(1).strip().lower()
                if "senior" in job_role or "lead" in job_role:
                    detected_role = 'senior'
                elif "junior" in job_role or "entry" in job_role:
                    detected_role = 'entry'
                else:
                    detected_role = 'mid'
            else:
                detected_role = 'mid'
            
            # Compute derived features
            salary_project_ratio = salary_expectation / (projects_count + 1e-6)  # Avoid division by zero
            exp_skill_interaction = experience_years * skill_match_score
            
            return {
                'Skill_Match': skill_match_score,
                'Experience (Years)': experience_years,
                'Education': education_level,
                'Certifications': certifications_count,
                'Projects Count': projects_count,
                'Job Role': detected_role,
                'Salary Expectation': salary_expectation,
                'Salary_Project_Ratio': salary_project_ratio,
                'Exp_Skill_Interaction': exp_skill_interaction,
                'Domain Expertise': expertise_score,
                'Combined_Text': standardized_resume[:2000],
                'resume_text': standardized_resume
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {
                'Skill_Match': 0,
                'Experience (Years)': 0,
                'Education': 0,
                'Certifications': 0,
                'Projects Count': 0,
                'Job Role': 'mid',
                'Salary Expectation': 0.0,
                'Salary_Project_Ratio': 0.0,
                'Exp_Skill_Interaction': 0.0,
                'Domain Expertise': 0.0,
                'Combined_Text': resume_text[:2000],
                'resume_text': resume_text[:1000] + "..."
            }
    
    def predict_score(self, features: Dict) -> float:
        """
        Predict AI score using a rule-based approach with domain context
        
        Args:
            features: Dictionary with extracted features
            
        Returns:
            Predicted score (0-100)
        """
        try:
            # Rule-based scoring dengan bobot domain-specific
            score = 0
            score += features['Skill_Match'] * 25  # 25% weight
            score += min(features['Experience (Years)'] / 10, 1.0) * 20  # 20% weight
            score += (features['Education'] / 4) * 15  # 15% weight
            score += min(features['Certifications'] / 5, 1.0) * 10  # 10% weight
            score += min(features['Projects Count'] / 10, 1.0) * 10  # 10% weight
            score += features['Domain Expertise'] * 15  # 15% weight
            score += (1 if features['Job Role'] == 'senior' else 0.5 if features['Job Role'] == 'mid' else 0) * 5  # 5% weight
            
            # Normalize to 0-100
            score = min(100, max(0, score))
            return score
        except Exception as e:
            logger.error(f"Error predicting score: {str(e)}")
            return 50.0  # Default score if prediction fails
    
    def score_by_criteria(self, overall_score: float, features: Dict) -> Dict[str, float]:
        """
        Break down overall score into individual criteria scores
        
        Args:
            overall_score: Overall AI score (0-100)
            features: Extracted features
            
        Returns:
            Dictionary mapping criteria to scores
        """
        scores = {}
        base_score = overall_score / 10
        
        for criterion, weight in self.criteria.items():
            if "skill" in criterion.lower():
                factor = features['Skill_Match']
                criterion_score = base_score * 0.7 + factor * 3
            elif "experience" in criterion.lower():
                factor = min(features['Experience (Years)'] / 10, 1.0)
                criterion_score = base_score * 0.6 + factor * 4
            elif "education" in criterion.lower():
                factor = features['Education'] / 4
                criterion_score = base_score * 0.8 + factor * 2
            elif "certification" in criterion.lower():
                factor = min(features['Certifications'] / 5, 1.0)
                criterion_score = base_score * 0.8 + factor * 2
            elif "project" in criterion.lower():
                factor = min(features['Projects Count'] / 10, 1.0)
                criterion_score = base_score * 0.7 + factor * 3
            elif "expertise" in criterion.lower() or "leadership" in criterion.lower():
                factor = features['Domain Expertise']
                criterion_score = base_score * 0.7 + factor * 3
            else:
                criterion_score = base_score
            
            weighted_score = min(10, max(1, criterion_score)) * (weight / 10)
            scores[criterion] = round(weighted_score, 2)
        
        return scores
    
    def detect_experience_level(self, features: Dict) -> str:
        """
        Detect experience level based on extracted features
        
        Args:
            features: Extracted features dictionary
            
        Returns:
            Experience level: 'entry', 'mid', 'senior', or 'expert'
        """
        experience_years = features['Experience (Years)']
        job_role = features['Job Role']
        skill_match = features['Skill_Match']
        expertise = features['Domain Expertise']
        
        if experience_years >= 10 or expertise >= 0.8:
            return 'expert'
        elif experience_years >= 7 or (job_role == 'senior' and expertise >= 0.5):
            return 'senior'
        elif experience_years >= 3 or (job_role == 'mid' and skill_match >= 0.5):
            return 'mid'
        else:
            return 'entry'
    
    def score_resume(self, resume_text: str, jd_text: Optional[str] = None) -> Dict:
        """
        Score a single resume using the rule-based model
        
        Args:
            resume_text: Raw resume text
            jd_text: Job description text (optional)
            
        Returns:
            Dictionary with scoring results
        """
        try:
            features = self.extract_features_from_resume(resume_text, jd_text)
            ai_score = self.predict_score(features)
            criteria_scores = self.score_by_criteria(ai_score, features)
            total_score = sum(criteria_scores.values())
            percentage = (total_score / self.max_score) * 100
            level = self.detect_experience_level(features)
            
            return {
                "scores": criteria_scores,
                "total_score": round(total_score, 2),
                "percentage": round(percentage, 2),
                "ai_score": round(ai_score, 2),
                "level": level,
                "features": features,
                "standardized_resume": features['resume_text']
            }
            
        except Exception as e:
            logger.error(f"Error scoring resume: {str(e)}")
            default_scores = {k: 5 * (v/10) for k, v in self.criteria.items()}
            return {
                "scores": default_scores,
                "total_score": sum(default_scores.values()),
                "percentage": 50.0,
                "ai_score": 50.0,
                "level": "mid",
                "features": {},
                "standardized_resume": resume_text[:1000] + "..."
            }
    
    def compare_resumes(self, resume_texts: List[str], jd_text: Optional[str] = None) -> Dict:
        """
        Score and compare multiple resumes
        
        Args:
            resume_texts: List of resume texts
            jd_text: Job description (optional, for context)
            
        Returns:
            Dictionary with ranking results
        """
        results = []
        
        for i, text in enumerate(resume_texts):
            scoring_result = self.score_resume(text, jd_text)
            
            results.append({
                "candidate_id": i + 1,
                **scoring_result,
                "text": text[:1000] + "...",
                "score_interpretation": {
                    k: self.scoring_guide[min(10, max(1, round(v / (self.criteria[k] / 10))))]
                    for k, v in scoring_result["scores"].items()
                }
            })
        
        results.sort(key=lambda x: (x["ai_score"], x["total_score"]), reverse=True)
        
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return {
            "ranking": results,
            "criteria": self.criteria,
            "max_score": self.max_score,
            "scoring_guide": self.scoring_guide
        }
