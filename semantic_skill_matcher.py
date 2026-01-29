"""
Semantic Skill Matcher for JD-Resume Analysis
==============================================

A modular, explainable system for matching Job Description requirements
against resume content using semantic similarity and context validation.

This module performs intelligent skill validation by:
1. Extracting context-rich skill evidence from resume sections
2. Using semantic embeddings to match JD requirements with resume content
3. Filtering skills based on real-world project/experience context
4. Providing explainable decisions with clear reasoning

Core Principle:
- Skills are validated only if they appear within project descriptions,
  work experience, or responsibility contexts
- Standalone "Skills" section entries are ignored unless corroborated
  by actual project/experience context
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Set
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class SkillValidationStatus(Enum):
    """Skill validation outcomes."""
    VALIDATED = "VALIDATED"  # Skill backed by project/experience context
    CONTEXTUAL_MISMATCH = "CONTEXTUAL_MISMATCH"  # Skill mentioned but not in context
    NOT_FOUND = "NOT_FOUND"  # Skill not present in resume
    IGNORED_STANDALONE = "IGNORED_STANDALONE"  # Skill only in Skills section
    WEAK_EVIDENCE = "WEAK_EVIDENCE"  # Minimal contextual support


class ContextType(Enum):
    """Types of contextual evidence in resume."""
    PROJECT = "PROJECT"
    WORK_EXPERIENCE = "WORK_EXPERIENCE"
    RESPONSIBILITY = "RESPONSIBILITY"
    EDUCATION = "EDUCATION"
    SKILLS_SECTION = "SKILLS_SECTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class SkillEvidence:
    """Represents evidence for a skill found in resume."""
    skill_name: str
    context_type: ContextType
    evidence_text: str
    confidence_score: float  # 0.0-1.0
    location: str = ""  # e.g., "Project: ML Pipeline" or "Experience: Company X"
    action_verbs: List[str] = field(default_factory=list)  # e.g., ["built", "implemented"]


@dataclass
class SkillValidationResult:
    """Result of validating a single skill."""
    skill_name: str
    jd_required: bool
    status: SkillValidationStatus
    resume_found: bool
    relevance_score: float  # 0.0-1.0
    evidence: List[SkillEvidence] = field(default_factory=list)
    reasoning: str = ""
    years_of_experience: Optional[int] = None
    is_production_experience: bool = False


@dataclass
class SkillMatchingReport:
    """Comprehensive skill matching analysis report."""
    overall_relevance_score: float  # 0.0-1.0
    total_jd_skills: int
    validated_skills: List[SkillValidationResult]
    ignored_skills: List[SkillValidationResult]
    missing_skills: List[SkillValidationResult]
    weak_skills: List[SkillValidationResult]
    resume_summary: str = ""
    recommendations: List[str] = field(default_factory=list)


class ResumeSectionExtractor:
    """
    Extracts and identifies different sections of a resume
    with proper context labeling.
    """

    # Patterns for common resume sections
    SECTION_PATTERNS = {
        "EXPERIENCE": r"(?:work\s+)?experience|employment|career history|professional\s+(?:background|summary)",
        "PROJECTS": r"projects?|portfolio|(?:key\s+)?achievements?",
        "EDUCATION": r"education|academic|degree|certification",
        "SKILLS": r"(?:technical\s+)?skills?|competencies?|expertise",
        "SUMMARY": r"(?:professional\s+)?summary?|objective|profile",
    }

    def __init__(self):
        """Initialize the section extractor."""
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in self.SECTION_PATTERNS.items()
        }

    def extract_sections(self, resume_text: str) -> Dict[str, str]:
        """
        Extract resume sections with their content.

        Args:
            resume_text: Full resume text

        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}
        lines = resume_text.split("\n")
        current_section = "UNCLASSIFIED"
        current_content = []

        for line in lines:
            # Check if line is a section header
            section_match = None
            for section_name, pattern in self.compiled_patterns.items():
                if pattern.search(line) and len(line.strip()) < 80:  # Headers are short
                    section_match = section_name
                    break

            if section_match:
                # Save previous section
                if current_content and current_section != "UNCLASSIFIED":
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = section_match
                current_content = []
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def identify_context_type(self, section_name: str) -> ContextType:
        """Map section name to context type."""
        mapping = {
            "EXPERIENCE": ContextType.WORK_EXPERIENCE,
            "PROJECTS": ContextType.PROJECT,
            "EDUCATION": ContextType.EDUCATION,
            "SKILLS": ContextType.SKILLS_SECTION,
        }
        return mapping.get(section_name, ContextType.UNKNOWN)


class ActionVerbDetector:
    """
    Identifies action verbs that indicate hands-on experience
    vs. passive skill listing.
    """

    # Action verbs grouped by strength
    STRONG_ACTION_VERBS = {
        "built", "implemented", "developed", "created", "deployed",
        "optimized", "designed", "architected", "engineered", "produced",
        "delivered", "launched", "published", "released"
    }

    MODERATE_ACTION_VERBS = {
        "worked", "contributed", "collaborated", "participated", "supported",
        "used", "applied", "managed", "maintained", "improved"
    }

    WEAK_ACTION_VERBS = {
        "familiar with", "knowledge of", "aware of", "basic understanding",
        "studied", "learned", "exposed to"
    }

    @classmethod
    def detect_verbs(cls, text: str) -> Tuple[List[str], float]:
        """
        Detect action verbs in text and score confidence.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (found_verbs, confidence_score)
        """
        text_lower = text.lower()
        found_verbs = []
        verb_strength = 0.0

        for verb in cls.STRONG_ACTION_VERBS:
            if re.search(r'\b' + verb + r'\b', text_lower):
                found_verbs.append(verb)
                verb_strength += 0.3

        for verb in cls.MODERATE_ACTION_VERBS:
            if re.search(r'\b' + verb + r'\b', text_lower):
                found_verbs.append(verb)
                verb_strength += 0.15

        for verb in cls.WEAK_ACTION_VERBS:
            if re.search(r'\b' + verb + r'\b', text_lower):
                found_verbs.append(verb)
                verb_strength += 0.05

        # Normalize confidence score
        confidence = min(verb_strength, 1.0)
        return found_verbs, confidence


class SemanticEmbeddingMatcher:
    """
    Uses semantic embeddings to match skills with resume content.
    """

    def __init__(self, embedding_model: Optional[FastEmbedEmbeddings] = None):
        """
        Initialize the semantic matcher.

        Args:
            embedding_model: FastEmbedEmbeddings instance (creates default if None)
        """
        self.embeddings = embedding_model or FastEmbedEmbeddings()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0

        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)

        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        # Normalize from [-1, 1] to [0, 1]
        return (cosine_sim + 1) / 2

    def find_best_match(
        self,
        skill_query: str,
        candidate_texts: List[str],
        threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching text for a skill.

        Args:
            skill_query: Skill to search for
            candidate_texts: Texts to search in
            threshold: Minimum similarity score

        Returns:
            Tuple of (best_match_text, similarity_score) or (None, 0.0)
        """
        best_match = None
        best_score = 0.0

        for text in candidate_texts:
            if not text:
                continue
            score = self.compute_similarity(skill_query, text)
            if score > best_score:
                best_score = score
                best_match = text

        if best_score >= threshold:
            return best_match, best_score
        return None, 0.0


class SkillContextValidator:
    """
    Validates skills by checking contextual evidence.
    Ignores standalone skill listings without project/experience support.
    """

    # Minimum thresholds
    SEMANTIC_MATCH_THRESHOLD = 0.65
    STRONG_MATCH_THRESHOLD = 0.75
    WEAK_MATCH_THRESHOLD = 0.55

    def __init__(self, embedding_matcher: Optional[SemanticEmbeddingMatcher] = None):
        """
        Initialize the validator.

        Args:
            embedding_matcher: SemanticEmbeddingMatcher instance
        """
        self.matcher = embedding_matcher or SemanticEmbeddingMatcher()

    def validate_skill(
        self,
        skill_name: str,
        resume_sections: Dict[str, str],
        section_contexts: Dict[str, ContextType]
    ) -> SkillValidationResult:
        """
        Validate a single skill across resume sections.

        Args:
            skill_name: Skill to validate
            resume_sections: Dict of section names to content
            section_contexts: Dict mapping section names to context types

        Returns:
            SkillValidationResult with validation details
        """
        result = SkillValidationResult(
            skill_name=skill_name,
            jd_required=False,  # Set by caller
            status=SkillValidationStatus.NOT_FOUND,
            resume_found=False,
            relevance_score=0.0
        )

        # Search in all sections except standalone Skills section
        evidence_list = []
        skills_section_found = False

        for section_name, section_content in resume_sections.items():
            if not section_content:
                continue

            context_type = section_contexts.get(section_name, ContextType.UNKNOWN)

            # Check for skill mention
            similarity, matched_text = self._find_skill_in_section(
                skill_name, section_content
            )

            if similarity > 0:
                # Found skill reference
                action_verbs, action_confidence = ActionVerbDetector.detect_verbs(
                    matched_text
                )

                # Combine semantic similarity with action verb confidence
                combined_score = (similarity + action_confidence) / 2

                evidence = SkillEvidence(
                    skill_name=skill_name,
                    context_type=context_type,
                    evidence_text=matched_text[:200],  # Truncate long evidence
                    confidence_score=combined_score,
                    location=section_name,
                    action_verbs=action_verbs
                )
                evidence_list.append(evidence)

                # Track if found only in Skills section
                if context_type == ContextType.SKILLS_SECTION:
                    skills_section_found = True

        # Determine validation status
        if not evidence_list:
            result.status = SkillValidationStatus.NOT_FOUND
            result.reasoning = f"'{skill_name}' not mentioned in resume."
            return result

        # Filter evidence: prioritize project/experience over skills section
        non_skills_evidence = [
            e for e in evidence_list
            if e.context_type != ContextType.SKILLS_SECTION
        ]

        if not non_skills_evidence and skills_section_found:
            # Skill only in standalone Skills section
            result.status = SkillValidationStatus.IGNORED_STANDALONE
            result.resume_found = True
            result.relevance_score = 0.0
            result.evidence = evidence_list
            result.reasoning = (
                f"'{skill_name}' found only in Skills section with no supporting "
                "project or work experience context. Requires real-world usage evidence."
            )
            return result

        # Use best evidence from project/experience context
        best_evidence = max(non_skills_evidence, key=lambda e: e.confidence_score)
        result.evidence = [best_evidence] + non_skills_evidence[1:]
        result.resume_found = True
        result.relevance_score = best_evidence.confidence_score

        # Determine final status
        if best_evidence.confidence_score >= self.STRONG_MATCH_THRESHOLD:
            result.status = SkillValidationStatus.VALIDATED
            context_name = best_evidence.context_type.value.lower()
            action_text = ", ".join(best_evidence.action_verbs) if best_evidence.action_verbs else "used"
            result.reasoning = (
                f"'{skill_name}' validated in {context_name} context "
                f"('{best_evidence.location}'). Evidence: {action_text} "
                f"(confidence: {best_evidence.confidence_score:.1%})"
            )
        elif best_evidence.confidence_score >= self.SEMANTIC_MATCH_THRESHOLD:
            result.status = SkillValidationStatus.VALIDATED
            result.reasoning = (
                f"'{skill_name}' found in {best_evidence.context_type.value.lower()} "
                f"with semantic match (confidence: {best_evidence.confidence_score:.1%})"
            )
        else:
            result.status = SkillValidationStatus.WEAK_EVIDENCE
            result.reasoning = (
                f"'{skill_name}' has weak contextual evidence "
                f"(confidence: {best_evidence.confidence_score:.1%}). "
                "Recommend manual verification."
            )

        return result

    def _find_skill_in_section(
        self,
        skill_name: str,
        section_content: str,
        context_window: int = 200
    ) -> Tuple[float, str]:
        """
        Find skill in section content and get matched context.

        Args:
            skill_name: Skill to find
            section_content: Section text to search
            context_window: Characters to include around match

        Returns:
            Tuple of (similarity_score, matched_context_text)
        """
        # First, try direct keyword match
        skill_pattern = re.compile(r'\b' + re.escape(skill_name) + r'\b', re.IGNORECASE)
        match = skill_pattern.search(section_content)

        if match:
            # Extract context window around match
            start = max(0, match.start() - context_window)
            end = min(len(section_content), match.end() + context_window)
            context_text = section_content[start:end].strip()

            # Use semantic similarity on the context
            similarity = self.matcher.compute_similarity(skill_name, context_text)
            return similarity, context_text

        # If no direct match, use semantic search on chunks
        chunks = self._chunk_text(section_content, chunk_size=300, overlap=100)
        best_match, best_score = self.matcher.find_best_match(
            skill_name,
            chunks,
            threshold=self.WEAK_MATCH_THRESHOLD
        )

        if best_match:
            return best_score, best_match
        return 0.0, ""

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class SemanticSkillMatcher:
    """
    Main orchestrator for semantic skill matching between JD and resume.
    Provides comprehensive, explainable validation results.
    """

    def __init__(
        self,
        embedding_matcher: Optional[SemanticEmbeddingMatcher] = None,
        validator: Optional[SkillContextValidator] = None
    ):
        """
        Initialize the semantic skill matcher.

        Args:
            embedding_matcher: SemanticEmbeddingMatcher instance
            validator: SkillContextValidator instance
        """
        self.matcher = embedding_matcher or SemanticEmbeddingMatcher()
        self.validator = validator or SkillContextValidator(self.matcher)
        self.extractor = ResumeSectionExtractor()

    def extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """
        Extract key skills from Job Description.

        Args:
            jd_text: Full JD text

        Returns:
            List of extracted skill names
        """
        # Common patterns for skill listings in JD
        patterns = [
            r"required\s+(?:skills?|competencies?)[:\s]+(.*?)(?:\n\n|\n(?=[A-Z])|$)",
            r"technical\s+skills?[:\s]+(.*?)(?:\n\n|\n(?=[A-Z])|$)",
            r"must have[:\s]+(.*?)(?:\n\n|\n(?=[A-Z])|$)",
            r"expertise in[:\s]+(.*?)(?:\n\n|\n(?=[A-Z])|$)",
        ]

        skills = set()
        for pattern in patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract individual skills (usually comma or newline separated)
                items = re.split(r'[,\n•\-\*]', match)
                for item in items:
                    skill = item.strip()
                    if skill and len(skill) > 2 and len(skill) < 50:
                        skills.add(skill)

        # Also try semantic extraction if no structured skills found
        if len(skills) < 5:
            skills.update(self._extract_semantic_skills(jd_text))

        return list(skills)

    def _extract_semantic_skills(self, text: str, top_n: int = 10) -> Set[str]:
        """
        Semantically extract likely skill entities from text.
        Looks for noun phrases following common skill indicators.
        """
        skill_keywords = {
            "python", "java", "javascript", "react", "angular", "vue",
            "sql", "mongodb", "postgresql", "aws", "azure", "gcp",
            "docker", "kubernetes", "jenkins", "git", "jira",
            "machine learning", "data science", "nlp", "cv",
            "rest api", "microservices", "agile", "scrum", "devops",
            "cloud", "ml", "ai", "deep learning", "tensorflow", "pytorch"
        }

        found_skills = set()
        text_lower = text.lower()

        for skill in skill_keywords:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.add(skill)

        return found_skills

    def analyze(
        self,
        jd_text: str,
        resume_text: str,
        jd_skills: Optional[List[str]] = None
    ) -> SkillMatchingReport:
        """
        Perform comprehensive semantic skill matching analysis.

        Args:
            jd_text: Job Description text
            resume_text: Resume text
            jd_skills: Optional pre-specified list of JD skills (otherwise extracted)

        Returns:
            SkillMatchingReport with detailed analysis
        """
        # Extract JD skills if not provided
        if not jd_skills:
            jd_skills = self.extract_skills_from_jd(jd_text)

        # Extract resume sections
        resume_sections = self.extractor.extract_sections(resume_text)
        section_contexts = {
            section: self.extractor.identify_context_type(section)
            for section in resume_sections.keys()
        }

        # Validate each skill
        validated_skills = []
        ignored_skills = []
        missing_skills = []
        weak_skills = []

        for skill in jd_skills:
            result = self.validator.validate_skill(skill, resume_sections, section_contexts)
            result.jd_required = True

            if result.status == SkillValidationStatus.VALIDATED:
                validated_skills.append(result)
            elif result.status == SkillValidationStatus.IGNORED_STANDALONE:
                ignored_skills.append(result)
            elif result.status == SkillValidationStatus.NOT_FOUND:
                missing_skills.append(result)
            elif result.status == SkillValidationStatus.WEAK_EVIDENCE:
                weak_skills.append(result)

        # Calculate overall relevance score
        overall_score = self._calculate_relevance_score(
            validated_skills, ignored_skills, missing_skills, weak_skills, len(jd_skills)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            validated_skills, ignored_skills, missing_skills, weak_skills
        )

        # Create summary
        resume_summary = self._create_resume_summary(resume_sections, validated_skills)

        return SkillMatchingReport(
            overall_relevance_score=overall_score,
            total_jd_skills=len(jd_skills),
            validated_skills=validated_skills,
            ignored_skills=ignored_skills,
            missing_skills=missing_skills,
            weak_skills=weak_skills,
            resume_summary=resume_summary,
            recommendations=recommendations
        )

    def _calculate_relevance_score(
        self,
        validated: List[SkillValidationResult],
        ignored: List[SkillValidationResult],
        missing: List[SkillValidationResult],
        weak: List[SkillValidationResult],
        total_jd_skills: int
    ) -> float:
        """
        Calculate overall relevance score (0.0-1.0).

        Scoring:
        - Validated: +100% per skill
        - Weak: +50% per skill
        - Ignored: 0% (no practical evidence)
        - Missing: 0%
        """
        if total_jd_skills == 0:
            return 0.0

        score = (
            len(validated) * 1.0 +
            len(weak) * 0.5
        ) / total_jd_skills

        return min(score, 1.0)

    @staticmethod
    def _generate_recommendations(
        validated: List[SkillValidationResult],
        ignored: List[SkillValidationResult],
        missing: List[SkillValidationResult],
        weak: List[SkillValidationResult]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if len(missing) > 0:
            missing_skills = [s.skill_name for s in missing[:3]]
            recommendations.append(
                f"Consider gaining hands-on experience with: {', '.join(missing_skills)}"
            )

        if len(ignored) > 0:
            ignored_skills = [s.skill_name for s in ignored[:3]]
            recommendations.append(
                f"Add project details for: {', '.join(ignored_skills)} "
                "(currently only in Skills section)"
            )

        if len(weak) > 0:
            recommendations.append(
                "Strengthen evidence for weakly-matched skills by adding "
                "specific project outcomes and metrics"
            )

        if len(validated) >= len(missing) + len(ignored):
            recommendations.append(
                "Strong project documentation. Consider expanding with specific metrics "
                "and technical depth in each experience"
            )

        return recommendations

    @staticmethod
    def _create_resume_summary(
        resume_sections: Dict[str, str],
        validated_skills: List[SkillValidationResult]
    ) -> str:
        """Create a brief summary of resume content."""
        summary_parts = []

        if "EXPERIENCE" in resume_sections and resume_sections["EXPERIENCE"]:
            exp_lines = len(resume_sections["EXPERIENCE"].split("\n"))
            summary_parts.append(f"Work experience documented across {exp_lines} lines")

        if "PROJECTS" in resume_sections and resume_sections["PROJECTS"]:
            proj_lines = len(resume_sections["PROJECTS"].split("\n"))
            summary_parts.append(f"Projects section with {proj_lines} lines of detail")

        skills_text = ", ".join([s.skill_name for s in validated_skills[:5]])
        summary_parts.append(f"Key validated skills: {skills_text}")

        return "; ".join(summary_parts) if summary_parts else "Resume structure not clearly identified"


# ==================== Output Formatting ====================

def format_report_as_dict(report: SkillMatchingReport) -> Dict:
    """
    Convert SkillMatchingReport to dictionary for JSON serialization.

    Args:
        report: SkillMatchingReport instance

    Returns:
        Dictionary representation
    """
    return {
        "overall_relevance_score": round(report.overall_relevance_score, 3),
        "summary": {
            "total_jd_skills": report.total_jd_skills,
            "validated_count": len(report.validated_skills),
            "ignored_count": len(report.ignored_skills),
            "missing_count": len(report.missing_skills),
            "weak_count": len(report.weak_skills),
        },
        "validated_skills": [
            {
                "skill": s.skill_name,
                "relevance_score": round(s.relevance_score, 2),
                "status": s.status.value,
                "reasoning": s.reasoning,
                "evidence_count": len(s.evidence),
            }
            for s in report.validated_skills
        ],
        "ignored_skills": [
            {
                "skill": s.skill_name,
                "status": s.status.value,
                "reasoning": s.reasoning,
            }
            for s in report.ignored_skills
        ],
        "missing_skills": [s.skill_name for s in report.missing_skills],
        "weak_skills": [
            {
                "skill": s.skill_name,
                "relevance_score": round(s.relevance_score, 2),
                "reasoning": s.reasoning,
            }
            for s in report.weak_skills
        ],
        "resume_summary": report.resume_summary,
        "recommendations": report.recommendations,
    }


def format_report_as_text(report: SkillMatchingReport) -> str:
    """
    Format SkillMatchingReport as human-readable text.

    Args:
        report: SkillMatchingReport instance

    Returns:
        Formatted text report
    """
    lines = [
        "=" * 80,
        "SEMANTIC SKILL MATCHING REPORT",
        "=" * 80,
        "",
        f"Overall Relevance Score: {report.overall_relevance_score:.1%}",
        f"Skills Analysis: {len(report.validated_skills)} validated, "
        f"{len(report.ignored_skills)} ignored, {len(report.missing_skills)} missing, "
        f"{len(report.weak_skills)} weak",
        "",
        "-" * 80,
        "VALIDATED SKILLS (Backed by Real Experience)",
        "-" * 80,
    ]

    if report.validated_skills:
        for skill in report.validated_skills:
            lines.append(f"\n✓ {skill.skill_name}")
            lines.append(f"  Relevance: {skill.relevance_score:.1%}")
            lines.append(f"  Reasoning: {skill.reasoning}")
    else:
        lines.append("(None)")

    lines.extend([
        "",
        "-" * 80,
        "IGNORED SKILLS (No Contextual Evidence)",
        "-" * 80,
    ])

    if report.ignored_skills:
        for skill in report.ignored_skills:
            lines.append(f"\n✗ {skill.skill_name}")
            lines.append(f"  Reason: {skill.reasoning}")
    else:
        lines.append("(None)")

    lines.extend([
        "",
        "-" * 80,
        "MISSING SKILLS (Not Mentioned in Resume)",
        "-" * 80,
    ])

    if report.missing_skills:
        lines.append("\n" + ", ".join([s.skill_name for s in report.missing_skills]))
    else:
        lines.append("(All JD skills present in resume)")

    lines.extend([
        "",
        "-" * 80,
        "WEAK EVIDENCE SKILLS (Minimal Contextual Support)",
        "-" * 80,
    ])

    if report.weak_skills:
        for skill in report.weak_skills:
            lines.append(f"\n⚠ {skill.skill_name}")
            lines.append(f"  Relevance: {skill.relevance_score:.1%}")
            lines.append(f"  Reasoning: {skill.reasoning}")
    else:
        lines.append("(None)")

    lines.extend([
        "",
        "-" * 80,
        "RESUME SUMMARY",
        "-" * 80,
        report.resume_summary,
    ])

    if report.recommendations:
        lines.extend([
            "",
            "-" * 80,
            "RECOMMENDATIONS",
            "-" * 80,
        ])
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    sample_jd = """
    We are looking for a Senior Python Developer with expertise in:
    - Python and FastAPI
    - PostgreSQL and Redis
    - Kubernetes and Docker
    - AWS cloud services
    - Machine Learning fundamentals
    
    Required Skills:
    - 5+ years of Python development
    - Experience with microservices
    - CI/CD pipelines (Jenkins, GitHub Actions)
    """

    sample_resume = """
    PROFESSIONAL EXPERIENCE
    
    Senior Software Engineer | TechCorp | 2021-Present
    - Architected and deployed a microservices platform using Python and FastAPI
    - Optimized PostgreSQL queries reducing latency by 40%
    - Implemented Redis caching layer for real-time data access
    - Led Kubernetes migration of legacy monolith (30% cost reduction)
    - Set up CI/CD pipelines using GitHub Actions and Jenkins
    
    Backend Developer | DataStartup | 2018-2021
    - Built Python REST APIs for customer analytics platform
    - Worked with AWS services (EC2, S3, Lambda)
    - Collaborated on machine learning feature engineering
    
    PROJECTS
    
    Real-time ML Pipeline
    - Deployed TensorFlow model serving with Docker containers
    - Integrated streaming data processing with Kubernetes jobs
    - Built monitoring dashboards for production metrics
    
    SKILLS
    Python, Java, JavaScript, PostgreSQL, MongoDB, Docker, Kubernetes,
    AWS, Azure, React, Node.js, FastAPI, Django, Redis, Elasticsearch
    """

    matcher = SemanticSkillMatcher()
    report = matcher.analyze(sample_jd, sample_resume)

    print(format_report_as_text(report))
    print("\n\nJSON Output:")
    import json
    print(json.dumps(format_report_as_dict(report), indent=2))
