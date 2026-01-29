"""
Integration module for Semantic Skill Matcher with Streamlit RAG pipeline.

Provides Streamlit-friendly functions to integrate semantic skill matching
into the existing JD-Resume analysis workflow.
"""

import json
from typing import Optional, Dict, List

from semantic_skill_matcher import (
    SemanticSkillMatcher,
    SkillMatchingReport,
    format_report_as_text,
    format_report_as_dict,
)


def run_semantic_analysis(
    jd_text: str,
    resume_text: str,
    jd_skills: Optional[List[str]] = None,
    output_format: str = "text"
) -> str:
    """
    Run semantic skill matching analysis and return formatted results.

    Args:
        jd_text: Job Description text
        resume_text: Resume text
        jd_skills: Optional list of skills to validate
        output_format: "text", "json", or "dict"

    Returns:
        Formatted analysis report
    """
    matcher = SemanticSkillMatcher()
    report = matcher.analyze(jd_text, resume_text, jd_skills)

    if output_format == "text":
        return format_report_as_text(report)
    elif output_format == "json":
        return json.dumps(format_report_as_dict(report), indent=2)
    elif output_format == "dict":
        return format_report_as_dict(report)
    else:
        return format_report_as_text(report)


def get_relevance_score(jd_text: str, resume_text: str) -> float:
    """
    Get overall relevance score quickly.

    Args:
        jd_text: Job Description text
        resume_text: Resume text

    Returns:
        Relevance score (0.0-1.0)
    """
    matcher = SemanticSkillMatcher()
    report = matcher.analyze(jd_text, resume_text)
    return report.overall_relevance_score


def get_validated_skills_list(jd_text: str, resume_text: str) -> List[str]:
    """
    Get list of validated skills.

    Args:
        jd_text: Job Description text
        resume_text: Resume text

    Returns:
        List of validated skill names
    """
    matcher = SemanticSkillMatcher()
    report = matcher.analyze(jd_text, resume_text)
    return [skill.skill_name for skill in report.validated_skills]


def get_ignored_skills_list(jd_text: str, resume_text: str) -> List[str]:
    """
    Get list of ignored skills (only in Skills section).

    Args:
        jd_text: Job Description text
        resume_text: Resume text

    Returns:
        List of ignored skill names
    """
    matcher = SemanticSkillMatcher()
    report = matcher.analyze(jd_text, resume_text)
    return [skill.skill_name for skill in report.ignored_skills]


def get_missing_skills_list(jd_text: str, resume_text: str) -> List[str]:
    """
    Get list of missing skills (not in resume).

    Args:
        jd_text: Job Description text
        resume_text: Resume text

    Returns:
        List of missing skill names
    """
    matcher = SemanticSkillMatcher()
    report = matcher.analyze(jd_text, resume_text)
    return [skill.skill_name for skill in report.missing_skills]


def create_streamlit_component(report: SkillMatchingReport) -> str:
    """
    Create Streamlit-formatted output for display.

    Args:
        report: SkillMatchingReport instance

    Returns:
        Formatted markdown string
    """
    md = f"""
## 📊 Semantic Skill Analysis Report

### Overall Relevance Score: **{report.overall_relevance_score:.0%}**

---

### Summary
- **Validated Skills**: {len(report.validated_skills)}
- **Ignored Skills**: {len(report.ignored_skills)}
- **Missing Skills**: {len(report.missing_skills)}
- **Weak Evidence Skills**: {len(report.weak_skills)}

---

### ✓ Validated Skills (Backed by Real Experience)
"""
    if report.validated_skills:
        for skill in report.validated_skills:
            md += f"\n- **{skill.skill_name}** ({skill.relevance_score:.0%})\n"
            md += f"  - {skill.reasoning}\n"
    else:
        md += "\n*None*\n"

    md += f"""
---

### ✗ Ignored Skills (No Contextual Evidence)

These skills appear only in the Skills section without supporting project or experience context.
"""
    if report.ignored_skills:
        for skill in report.ignored_skills:
            md += f"\n- **{skill.skill_name}**\n"
            md += f"  - {skill.reasoning}\n"
    else:
        md += "\n*None*\n"

    md += f"""
---

### ⚠️ Missing Skills

Not mentioned in resume:
"""
    if report.missing_skills:
        md += f"\n{', '.join([s.skill_name for s in report.missing_skills])}\n"
    else:
        md += "\n*All JD skills are present in resume*\n"

    md += f"""
---

### 💡 Resume Summary
{report.resume_summary}

---

### 🎯 Recommendations
"""
    if report.recommendations:
        for i, rec in enumerate(report.recommendations, 1):
            md += f"\n{i}. {rec}\n"
    else:
        md += "\n*No specific recommendations*\n"

    return md


if __name__ == "__main__":
    # Example for testing
    print("Semantic Skill Matcher Integration Module")
    print("Ready for Streamlit integration")
