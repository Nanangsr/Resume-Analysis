from typing import List, Dict, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import re
import logging
import streamlit as st
from utils.resume_standardizer import ResumeStandardizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ResumeScorer:
    def __init__(self, criteria: Dict[str, int]):
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b", 
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.standardizer = ResumeStandardizer()
        self.criteria = criteria
        self.max_score = sum(self.criteria.values())
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
        self.level_expectations = {
            "junior": {
                "Technical Skills": (3, 6),
                "Work Experience": (2, 5),
                "Leadership": (1, 4)
            },
            "mid": {
                "Technical Skills": (5, 8),
                "Work Experience": (4, 7),
                "Leadership": (3, 6)
            },
            "senior": {
                "Technical Skills": (7, 10),
                "Work Experience": (6, 9),
                "Leadership": (5, 8)
            }
        }
        
    def _extract_score(self, llm_response: str) -> float:
        """Extract numerical score from LLM response with improved parsing"""
        try:
            # More robust score extraction
            response = llm_response.strip().lower()
            
            # Look for explicit score indicators
            if "score:" in response:
                score_part = response.split("score:")[1].strip()
                numbers = re.findall(r'\d+\.?\d*', score_part)
                if numbers:
                    return float(numbers[0])
            
            # Look for numerical values in the entire response
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return max(1, min(10, score))
            
            # Look for textual indicators
            if any(word in response for word in ["excellent", "outstanding", "9", "10"]):
                return 9.0
            elif any(word in response for word in ["good", "strong", "7", "8"]):
                return 7.5
            elif any(word in response for word in ["average", "moderate", "4", "5", "6"]):
                return 5.0
            else:
                return 3.0
                
        except Exception as e:
            logger.error(f"Score extraction error: {str(e)}")
            return 5.0  # Safe default score
        
    def detect_experience_level(self, resume_text: str) -> str:
        """Detect candidate level based on resume content"""
        prompt = ChatPromptTemplate.from_template(
            """Analyze this resume and determine the candidate's experience level:
            - 'junior': 0-3 years experience
            - 'mid': 4-7 years experience
            - 'senior': 8+ years experience
            
            Resume:
            {resume_text}
            
            Return ONLY one of: ['junior', 'mid', 'senior']"""
        )
        
        chain = prompt | self.llm
        level = chain.invoke({"resume_text": resume_text[:3000]}).content.lower()
        return level if level in ['junior', 'mid', 'senior'] else 'mid'
    
    def score_resume(self, resume_text: str) -> Dict:  # Ubah return type ke Dict
        """Score a resume with level-adjusted expectations"""
        try:
            # Pertama standardisasi resume
            standardized_resume = self.standardizer.standardize_resume(resume_text)
            
            # Kemudian deteksi level
            level = self.standardizer.detect_resume_level(standardized_resume)
            
            scores = {}
            total_score = 0
            
            for criterion, weight in self.criteria.items():
                # Dapatkan range expected untuk level ini
                min_exp, max_exp = self.level_expectations.get(level, {}).get(criterion, (1, 10))
                
                prompt = ChatPromptTemplate.from_template(
                    """Evaluate this resume for {criterion} (Weight: {weight}):
                    Candidate Level: {level}
                    Expected Range for Level: {min_exp}-{max_exp}
                    
                    Evaluation Scale:
                    1-3 = Below expectations for level
                    4-6 = Meets basic expectations
                    7-8 = Exceeds expectations
                    9-10 = Outstanding for level
                    
                    Resume Excerpt:
                    {resume_excerpt}
                    
                    Provide ONLY the numerical score (1-10)"""
                )
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "criterion": criterion,
                    "weight": weight,
                    "level": level,
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "resume_excerpt": standardized_resume[:1500]
                })
                
                raw_score = self._extract_score(response.content)
                weighted_score = raw_score * (weight / 10)
                
                scores[criterion] = weighted_score
                total_score += weighted_score
            
            percentage = (total_score / self.max_score) * 100
            
            # Kembalikan sebagai dictionary saja
            return {
                "scores": scores,
                "total_score": total_score,
                "percentage": percentage,
                "level": level,
                "standardized_resume": standardized_resume
            }
            
        except Exception as e:
            logger.error(f"Error scoring resume: {str(e)}")
            # Return default scores if error occurs
            return {
                "scores": {k: 5 * (v/10) for k, v in self.criteria.items()},
                "total_score": 50,
                "percentage": 50.0,
                "level": "mid",
                "standardized_resume": resume_text[:1000] + "..." 
            }
    
    def compare_resumes(self, resume_texts: List[str], jd_text: str = None) -> Dict:
        """Score and compare multiple resumes with enhanced reliability"""
        results = []
        
        for i, text in enumerate(resume_texts):
            # Sekarang score_resume mengembalikan dict, bukan tuple
            scoring_result = self.score_resume(text)
            
            results.append({
                "candidate_id": i+1,
                **scoring_result,  # Unpack semua nilai dari score_resume
                "text": text[:1000] + "...",
                "score_interpretation": {
                    k: self.scoring_guide[round(v / (self.criteria[k] / 10))]
                    for k, v in scoring_result["scores"].items()
                }
            })
        
        # Rank candidates by total_score
        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Add ranking position
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return {
            "ranking": results,
            "criteria": self.criteria,
            "max_score": self.max_score,
            "scoring_guide": self.scoring_guide
        }
def _adjust_for_level(self, score: float, min_exp: int, max_exp: int) -> float:
        """Normalize score based on level expectations"""
        # Jika score di bawah minimum untuk levelnya
        if score < min_exp:
            return max(1, score * 0.7)  # Penalize more for underperforming
        # Jika score di range expected
        elif min_exp <= score <= max_exp:
            return score * 1.1  # Slight bonus for meeting expectations
        # Jika di atas expected range
        else:
            return min(10, score * 1.3)  # Reward for exceeding level expectations
        
def score_and_rank_candidates(resume_texts: List[str], jd_text: Optional[str] = None, criteria: Optional[Dict[str, int]] = None) -> Dict:
    """Score and rank candidates based on criteria with improved reliability"""
    if not criteria:
        criteria = {
            "Technical Skills": 8,
            "Education": 6,
            "Work Experience": 9,
            "Leadership": 7,
            "Communication": 6
        }
    
    scorer = ResumeScorer(criteria)
    results = scorer.compare_resumes(resume_texts, jd_text)
    
    # Prepare detailed analysis
    analysis_prompt = ChatPromptTemplate.from_template(
        """Analyze these candidates based on their scores:

        Scoring Guide:
        1-3 = Tidak memenuhi
        4-6 = Memenuhi sebagian
        7-8 = Memenuhi dengan baik
        9-10 = Melebihi ekspektasi

        Job Description:
        {jd_text}

        Candidate Scores:
        {ranking_summary}

        Provide analysis with:
        1. Top 3 candidates' strengths
        2. Notable weaknesses
        3. Best fit for role
        4. Training recommendations for lower-scored candidates

        Use Indonesian language for the analysis.
        """
    )
    
    # Format ranking information with interpretations
    ranking_summary = "\n\n".join(
        f"Candidate #{r['rank']} (Overall: {r['total_score']:.1f}/{scorer.max_score}, {r['percentage']:.1f}%)\n"
        + "\n".join(
            f"- {k}: {v:.1f} ({r['score_interpretation'][k]})"
            for k, v in r['scores'].items()
        )
        for r in results['ranking'][:5]  # Show top 5 for analysis
    )
    
    chain = analysis_prompt | scorer.llm
    analysis = chain.invoke({
        "jd_text": jd_text or "No specific job description provided",
        "ranking_summary": ranking_summary
    }).content
    
    # Add analysis to results
    results["analysis"] = analysis
    
    return results