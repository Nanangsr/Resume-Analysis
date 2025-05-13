from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from typing import List, Tuple
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ResumeStandardizer:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b",  # More cost-effective model
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
        self._init_prompts()
        
    def _init_prompts(self):
        """Initialize and cache prompt templates"""
        self.standardization_prompt = ChatPromptTemplate.from_template(
            """Transform this resume into a perfect ATS-friendly format following EXACTLY these rules:
            
            [STRUCTURE RULES]
            1. MANDATORY SECTIONS (in this order):
               - SUMMARY: 3-5 sentence professional overview
               - EXPERIENCE: Reverse chronological, MM/YYYY dates
               - EDUCATION: Degrees only, MM/YYYY
               - SKILLS: Grouped by category (Technical, Soft, etc.)
               - (Optional) CERTIFICATIONS, PROJECTS if applicable
            
            2. FORMATTING:
               - Bullet points only (no paragraphs)
               - Start each bullet with strong action verb
               - Quantify achievements (e.g., "Increased X by 30%")
               - Remove: pronouns, personal info (address, birthdate)
               - Company names in bold: **Company**
               - Job titles in italics: *Title*
            
            3. CONTENT:
               - Keep only professional information
               - Remove: photos, tables, graphics
               - Standardize degree names (BSc not Bachelor of Science)
            
            Input Resume:
            {resume_text}
            
            Output ONLY the perfectly formatted resume text, NO additional commentary.
            """
        )
        
        self.level_detection_prompt = ChatPromptTemplate.from_template(
            """Analyze resume experience and determine level (STRICT RULES):
            
            [LEVEL CRITERIA]
            Junior:
            - 0-3 years total experience
            - Entry-level job titles
            - No team leadership
            - Education may be highlighted
            
            Mid:
            - 4-7 years experience
            - Some project/team leadership
            - Moderate achievements
            - Specialized skills
            
            Senior:
            - 8+ years experience
            - Clear career progression
            - Significant leadership
            - Strategic impact evidence
            
            [ANALYSIS]
            Resume Excerpt:
            {resume_excerpt}
            
            Return ONLY the exact level keyword: junior | mid | senior
            """
        )

    def standardize_resume(self, resume_text: str) -> str:
        """Optimized resume standardization with validation"""
        try:
            if not resume_text or len(resume_text.strip()) < 50:
                raise ValueError("Resume text too short")
                
            # Pre-process text
            cleaned_text = self._preprocess_text(resume_text)
            
            chain = self.standardization_prompt | self.llm
            result = chain.invoke({"resume_text": cleaned_text}).content
            
            # Post-process and validate
            return self._validate_standardized_resume(result)
            
        except Exception as e:
            logger.error(f"Standardization failed: {str(e)}")
            return self._fallback_format(resume_text)

    def detect_resume_level(self, resume_text: str) -> str:
        """More reliable level detection with fallback"""
        try:
            excerpt = self._extract_relevant_excerpt(resume_text)
            chain = self.level_detection_prompt | self.llm
            response = chain.invoke({"resume_excerpt": excerpt}).content.lower()
            
            # Validate response
            if response not in {'junior', 'mid', 'senior'}:
                raise ValueError(f"Invalid level detected: {response}")
                
            return response
            
        except Exception as e:
            logger.warning(f"Level detection failed: {str(e)}")
            return self._estimate_level_fallback(resume_text)

    def standardize_multiple(self, resume_texts: List[str]) -> Tuple[List[str], List[str]]:
        """Batch processing with progress tracking"""
        standardized = []
        errors = []
        
        for text in resume_texts:
            try:
                standardized.append(self.standardize_resume(text))
            except Exception as e:
                standardized.append("")
                errors.append(str(e))
                
        return standardized, errors
    
    def _preprocess_text(self, text: str) -> str:
        """Clean input text before processing"""
        # Remove multiple spaces, weird characters
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-ASCII characters
        return text.encode('ascii', 'ignore').decode('ascii')

    def _validate_standardized_resume(self, text: str) -> str:
        """Check if output meets minimum requirements"""
        required_sections = {'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS'}
        present_sections = {s for s in required_sections if s in text}
        
        if len(present_sections) < 3:
            raise ValueError("Missing required sections")
            
        # Check bullet point usage
        if text.count('•') < 5 and text.count('-') < 5:
            raise ValueError("Insufficient bullet points")
            
        return text

    def _extract_relevant_excerpt(self, text: str) -> str:
        """Extract most relevant portion for level detection"""
        # Prioritize experience section
        experience_match = re.search(r'EXPERIENCE(.+?)(?=\n[A-Z]{3,}|$)', text, re.DOTALL|re.IGNORECASE)
        if experience_match:
            return experience_match.group(1)[:2000]
        return text[:1500]

    def _estimate_level_fallback(self, text: str) -> str:
        """Simple fallback level estimation"""
        year_patterns = r'(\d{4}[-–]\d{4}|\d{4}[^\d]\w+ \d{4})'
        matches = re.findall(year_patterns, text)
        total_years = min(len(matches) * 2, 15)  # Estimate 2 years per position
        
        if total_years <= 3:
            return 'junior'
        elif total_years <= 7:
            return 'mid'
        return 'senior'

    def _fallback_format(self, text: str) -> str:
        """Basic formatting when LLM fails"""
        sections = [
            "SUMMARY:\n[Extracted professional summary]",
            "\nEXPERIENCE:\n" + re.sub(r'(?<=\n)(?=\S)', '• ', text),
            "\nSKILLS:\n[Extracted skills]"
        ]
        return '\n'.join(sections)