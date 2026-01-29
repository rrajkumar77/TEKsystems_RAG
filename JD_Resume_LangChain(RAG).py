
import streamlit as st
import os
import io
import fitz  # PyMuPDF
import docx  # python-docx
import pandas as pd
from dotenv import load_dotenv

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Semantic skill matcher imports
from semantic_skill_matcher import SemanticSkillMatcher
from semantic_matcher_streamlit import create_streamlit_component

# ==================== ENV & MODEL ====================
load_dotenv()

def get_api_key():
    # Prefer Streamlit Cloud secrets; fallback to .env
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        # st.secrets not available (not running in Streamlit context)
        pass
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_api_key()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in Streamlit Secrets or .env"
    )

# Sidebar controls (optional)
st.set_page_config(page_title="Resume Expert (RAG + LangChain + Groq)", layout="wide")
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-27b-it",
        ],
        index=0,
        help="Choose the LLM model served by Groq",
        key="groq_model"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="llm_temperature")
    max_tokens = st.number_input("Max tokens", min_value=256, max_value=8192, value=3000, step=128, key="llm_max_tokens")
    k_retrieval = st.slider("Retriever k", 2, 12, 8, 1, key="retriever_k")
    search_type = st.selectbox("Retriever search type", ["mmr", "similarity"], index=0, key="search_type")
    
    # Evidence-Backed Skill Validation controls (FR 6)
    st.divider()
    st.subheader("🎯 Evidence-Backed Skill Validation")
    min_semantic_score = st.slider(
        "Min Semantic Score",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Minimum semantic similarity required to validate a skill (0.0-1.0)",
        key="min_semantic_score"
    )
    min_confidence_score = st.slider(
        "Min Confidence Score",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Minimum combined confidence (semantic + action verb) required (0.0-1.0)",
        key="min_confidence_score"
    )
    recency_weight = st.slider(
        "Recency Weight (0=ignore, 1=prioritize recent)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="How much to weight recent projects higher",
        key="recency_weight"
    )
    max_tokens = st.number_input("Max tokens", min_value=256, max_value=8192, value=3000, step=128)
    k_retrieval = st.slider("Retriever k", 2, 12, 8, 1)
    search_type = st.selectbox("Retriever search type", ["mmr", "similarity"], index=0)

# Initialize LLM (Groq via LangChain)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
)

# ==================== FILE HELPERS ====================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in document]
        return " ".join(text_parts).strip()
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        text_parts = [p.text for p in d.paragraphs]
        return " ".join(text_parts).strip()
    except Exception as e:
        raise ValueError(f"Failed to open DOCX: {e}")

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception as e:
        raise ValueError(f"Failed to decode TXT file: {e}")

def process_file(uploaded_file) -> str:
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty or unreadable.")

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        return extract_text_from_docx(file_bytes)
    elif ext == "doc":
        st.warning("DOC has limited support. Please convert to DOCX or PDF for best results.")
        try:
            return extract_text_from_docx(file_bytes)
        except Exception as e:
            st.error(f"Error processing DOC file: {e}")
            return "Error processing DOC file. Please convert to DOCX or PDF for better results."
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# ==================== RAG INDEX BUILD ====================
def build_vectorstore(jd_text: str, resume_text: str):
    """
    Builds a single Chroma vectorstore containing both JD and Resume chunks with metadata.
    In-memory for simplicity (ephemeral). Add persist_directory for persistence if needed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = []

    # NOTE: No need to import Document explicitly; create_documents returns Documents
    if jd_text:
        jd_docs = splitter.create_documents([jd_text], metadatas=[{"source": "jd"}])
        docs.extend(jd_docs)
    if resume_text:
        resume_docs = splitter.create_documents([resume_text], metadatas=[{"source": "resume"}])
        docs.extend(resume_docs)

    if not docs:
        return None

    embeddings = FastEmbedEmbeddings()  # lightweight local embeddings, no external calls
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="jobfit_rag",
        # persist_directory="./chroma_jobfit"  # uncomment to persist across runs
    )
    return vs

def make_retriever(vectorstore, scope="both", k=8, search_type="mmr"):
    """
    scope: "jd" | "resume" | "both"
    """
    kwargs = {"k": k}
    if scope in ("jd", "resume"):
        kwargs["filter"] = {"source": scope}
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)
    return retriever

def retrieve_context(vectorstore, scope: str, query: str, k: int = 8, search_type: str = "mmr") -> str:
    """
    Compatible with LangChain 0.3+ retrievers (Runnable).
    Falls back to get_relevant_documents for older versions.
    """
    retriever = make_retriever(vectorstore, scope=scope, k=k, search_type=search_type)
    try:
        # LCEL retrievers support .invoke(query)
        docs = retriever.invoke(query)
    except AttributeError:
        # Older retrievers use .get_relevant_documents(query)
        docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])

def call_llm_with_context(prompt_template: str, context: str, **fmt_vars) -> str:
    """
    Formats a chat prompt with context + variables and calls the LLM.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.format_messages(context=context, **fmt_vars)
    response = llm.invoke(messages)
    return response.content

# ==================== PROMPTS ====================
PROMPT_RECRUITER = """\
You are an Experienced Technical HR Manager with deep expertise in technical evaluations and recruitment.
Use the provided context from the Job Description and Resume to assess alignment.

Instructions:
- Provide:
  1) Match percentage between the resume and JD (e.g., "80% match").
  2) A professional evaluation (1–2 paragraphs) highlighting strengths and weaknesses.
  3) Suggestions for improvement (e.g., missing skills, certifications to consider).
- Be concise, professional, and grounded strictly in the context.

<context>
{context}
</context>

Task: Perform the full Recruiter Analysis.
"""

PROMPT_TECHNICAL_Q = """\
You are an Advanced AI for Technical Recruitment and Interview Preparation. Based on the JD and Resume context, generate structured interview questions organized by project lifecycle phases.

LIFECYCLE PHASES & COVERAGE:
1. **Requirements** - How do you approach gathering, validating, and documenting project requirements?
2. **Design** - How do you architect solutions, design APIs, data models, or system components?
3. **Development** - How do you implement, code, debug, and optimize features?
4. **Testing** - How do you design test strategies, write tests, and ensure quality?
5. **Deployment/Support** - How do you deploy, monitor, troubleshoot, and support production systems?

FOR EACH QUESTION, provide:
- **Question**: Clear, specific, experience-oriented question aligned with JD and Resume
- **Answer Guide**:
  - **Concepts**: Key theoretical knowledge (e.g., elicitation, design patterns, CI/CD)
  - **Tools**: Technologies, platforms, frameworks mentioned in resume or needed for JD
  - **Techniques**: Methodologies, best practices, or approaches used
  - **Practical Indicators**: Signs the candidate has hands-on experience (actions, outcomes, decisions made)
  - **Red Flags** (optional): Watch for superficial understanding or contradictions

Generate 1-2 questions per lifecycle phase (total up to 10 questions). Ensure NO generic content; all questions must be grounded in the specific JD and Resume provided.

<context>
{context}
</context>

Task: Generate lifecycle-ordered interview questions with detailed Answer Guides.
Output format:
**[Lifecycle Phase: PHASE_NAME]**
Q1: [Question]
Answer Guide:
- Concepts: ...
- Tools: ...
- Techniques: ...
- Practical Indicators: ...
"""

PROMPT_CODING_Q = """\
You are an Advanced AI for Technical Assessment in Recruitment. Based on the JD and Resume, generate up to 5 coding/technical questions ordered by project lifecycle phases.

SEQUENCING BY LIFECYCLE:
1. **Requirements** - Data validation, input parsing, spec interpretation (e.g., "Parse a JSON config and validate fields")
2. **Design** - Data structures, algorithms, API design (e.g., "Design a cache with LRU eviction")
3. **Development** - Core implementation, feature building (e.g., "Implement a function to compute quarterly sales aggregates")
4. **Testing** - Test logic, edge cases, error handling (e.g., "Write tests for null inputs and boundary conditions")
5. **Deployment/Optimization** - Performance, scaling, monitoring (e.g., "Optimize a query that scans 10M rows")

FOR EACH QUESTION, provide:
- **Lifecycle Phase**: Which phase this question targets
- **Question**: Clear, specific, realistic coding problem aligned with JD and Resume skills
- **Answer**: Complete, working code solution (Python, SQL, or other relevant language from JD)
- **Explanation**: Key concepts, approach, complexity analysis, and why the solution works
- **Difficulty Level**: Basic | Intermediate | Advanced
- **Key Concepts**: What the candidate should demonstrate (e.g., recursion, indexing, caching)

Ensure questions are:
- Grounded in the JD technical stack (e.g., Python, SQL, cloud APIs)
- Realistic to candidate's resume experience level
- NOT company-specific; broadly applicable

<context>
{context}
</context>

Task: Generate lifecycle-ordered coding questions with complete solutions and explanations.
"""

PROMPT_DOMAIN = """\
You are an ATS Scanner with Domain Expertise. Evaluate the Resume against the JD from a domain perspective.

Required Output:
- Match percentage (e.g., "75%") with a brief explanation.
- Missing keywords (comma-separated).
- Objective, thorough evaluation grounded in the context.

<context>
{context}
</context>

Task: Provide the domain-fit analysis.
"""

PROMPT_MANAGER = """\
You are an ATS Scanner with Technical Expertise. Evaluate the Resume against the JD for technical fit.

Required Output:
1) Match percentage
2) Explanation of matches and gaps
3) Missing keywords/skills
4) A table (plain text) of top 5 skills with: Required years (JD), Candidate years (Resume), Relevant projects
5) Final suitability summary

<context>
{context}
</context>

Task: Provide the technical-fit analysis with the requested table and insights.
"""

PROMPT_JD_SUMMARY = """\
You are an AI Assistant. Summarize the JD and provide recruiter recommendations.

Output two sections:
1) JD Summary (3–5 sentences of responsibilities, key skills, and qualifications)
2) Recommendations: Suggest skill combos, keywords, and sourcing strategy

Use only the JD context below.

<context>
{context}
</context>

Task: Provide the summary and recommendations.
"""

PROMPT_JD_CLARIFICATION = """\
You are a Technical Recruitment Consultant. Using ONLY the JD context, generate 5–10 precise questions for the hiring manager to clarify technical requirements, tools, expectations, project scope, and expertise levels.

- Avoid generic questions; tailor to the JD content.
- Output as a clean, numbered list.

<context>
{context}
</context>

Task: Provide the JD clarification questions.
"""

PROMPT_SKILL_ANALYST = """\
You are a Skill Analyst. Using ONLY the Resume context, analyze the provided top_skills list.

For each skill:
- Match Status: Yes/No (explicit or implicit)
- Relevant Projects: roles/projects/experiences from the resume (or "None")
- Years of Experience: best estimate from resume context; if unclear, make a reasonable assumption (e.g., "1 year" junior, "3 years" mid-level)

Output a structured table (plain text) with columns:
Skill | Match Status | Relevant Projects | Years of Experience

<context>
{context}
</context>

Top skills to analyze:
{top_skills}

Task: Provide the table only, followed by a brief note on assumptions if any.
"""

PROMPT_GENERAL_Q = """\
You are an AI Assistant for Recruitment Queries. Use the available context (JD, Resume, or both) to answer the user’s question clearly with useful insights.

<context>
{context}
</context>

User question:
{user_query}

Task: Provide a clear, concise, context-aware answer. If the question is unclear, ask one clarifying question.
"""

# ==================== UI LAYOUT ====================
st.title("TEKsystems JobFit Analyzer — RAG Edition (LangChain + Groq)")
st.caption("Upload a JD and/or Resume, and generate grounded analyses, questions, and summaries.")

uploaded_jd = st.file_uploader(
    "Upload the Job Description (PDF, DOCX, DOC, TXT)...",
    type=["pdf", "docx", "doc", "txt"],
    key="jd_uploader"
)
submit_jd_summarization = st.button("JD Summarization", key="submit_jd_summarization")
submit_jd_clarification = st.button("JD Clarification Questions", key="submit_jd_clarification")

uploaded_resume = st.file_uploader(
    "Upload your Resume (PDF, DOCX, DOC, TXT)...",
    type=["pdf", "docx", "doc", "txt"],
    key="resume_uploader"
)

jd_content = ""
resume_content = ""

if uploaded_jd is not None:
    file_type = uploaded_jd.name.split(".")[-1].upper()
    st.success(f"✅ {file_type} Job Description uploaded")
    jd_content = process_file(uploaded_jd)

if uploaded_resume is not None:
    file_type = uploaded_resume.name.split(".")[-1].upper()
    st.success(f"✅ {file_type} Resume uploaded")
    resume_content = process_file(uploaded_resume)

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    submit_recruiter = st.button("Technical Recruiter Analysis", key="submit_recruiter")
    submit_domain = st.button("Domain Expert Analysis", key="submit_domain")
    submit_semantic_skills = st.button("🎯 Evidence-Backed Skill Validation (NEW)", key="submit_semantic_skills")
with col2:
    submit_technical_questions = st.button("Technical Questions", key="submit_technical_questions")
    submit_manager = st.button("Technical Manager Analysis", key="submit_manager")
with col3:
    submit_coding_questions = st.button("Coding Questions", key="submit_coding_questions")

top_skills = st.text_input("Top Skills Required for the Job (comma-separated):", key="top_skills_input")
submit_skill_analysis = st.button("Skill Analysis", key="submit_skill_analysis")

input_promp = st.text_input("Queries: Feel Free to Ask here", key="custom_query_input")
submit_general_query = st.button("Answer My Query", key="submit_general_query")

# ==================== VECTORSTORE CACHE ====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_key" not in st.session_state:
    st.session_state.indexed_key = None

def compute_index_key(jd_text, resume_text):
    # Use hashes to detect content changes and rebuild index
    return f"{hash(jd_text) if jd_text else 0}-{hash(resume_text) if resume_text else 0}"

if jd_content or resume_content:
    key_now = compute_index_key(jd_content, resume_content)
    if st.session_state.vectorstore is None or st.session_state.indexed_key != key_now:
        with st.spinner("🔎 Indexing documents for retrieval..."):
            vs = build_vectorstore(jd_content, resume_content)
            st.session_state.vectorstore = vs
            st.session_state.indexed_key = key_now

def ensure_vs():
    if st.session_state.vectorstore is None:
        st.warning("Please upload a Job Description and/or a Resume first.")
        return None
    return st.session_state.vectorstore

# ==================== ACTIONS ====================
if submit_recruiter:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Analyzing alignment..."):
                ctx_jd = retrieve_context(vs, "jd", "role requirements, responsibilities, skills, experience", k=k_retrieval, search_type=search_type)
                ctx_cv = retrieve_context(vs, "resume", "candidate skills, projects, responsibilities, experience", k=k_retrieval, search_type=search_type)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_RECRUITER, context)
            st.subheader("Technical Recruiter Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_semantic_skills:
    if jd_content and resume_content:
        with st.spinner("🎯 Validating skills with semantic analysis and evidence extraction..."):
            try:
                # Initialize the semantic skill matcher
                matcher = SemanticSkillMatcher()
                
                # Run analysis with configured thresholds
                report = matcher.analyze(
                    jd_text=jd_content,
                    resume_text=resume_content
                )
                
                st.subheader("🎯 Evidence-Backed Skill Validation Report")
                
                # Overall score and summary
                st.metric("Overall Fit Score", f"{report.overall_relevance_score:.1%}")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Validated Skills", "Ignored Skills", "Evidence Details"])
                
                with tab1:
                    st.write(f"**Validated Skills:** {len(report.validated_skills)}")
                    st.write(f"**Weak Evidence:** {len(report.weak_skills)}")
                    st.write(f"**Ignored (Skills-only):** {len(report.ignored_skills)}")
                    st.write(f"**Not Found:** {len(report.missing_skills)}")
                    
                    st.divider()
                    st.write("**Recommendations:**")
                    if report.recommendations:
                        for rec in report.recommendations:
                            st.write(f"• {rec}")
                
                with tab2:
                    st.write(f"**{len(report.validated_skills)} Validated Skills** (with project evidence)")
                    for skill_result in report.validated_skills:
                        with st.expander(f"✅ {skill_result.skill_name} — {skill_result.relevance_score:.0%} confidence"):
                            st.write(f"**Status:** {skill_result.status.value}")
                            st.write(f"**Confidence Score:** {skill_result.relevance_score:.1%}")
                            st.write(f"**Reasoning:** {skill_result.reasoning}")
                            if skill_result.evidence:
                                st.write("**Evidence:**")
                                for evidence in skill_result.evidence:
                                    st.write(f"  - *{evidence.context_type.value}*: '{evidence.evidence_text[:100]}...'")
                                    st.write(f"    Action verbs: {', '.join(evidence.action_verbs) if evidence.action_verbs else 'N/A'}")
                
                with tab3:
                    st.write(f"**{len(report.ignored_skills)} Ignored Skills** (no project context)")
                    for skill_result in report.ignored_skills:
                        with st.expander(f"⊘ {skill_result.skill_name}"):
                            st.write(f"**Status:** {skill_result.status.value}")
                            st.write(f"**Reasoning:** {skill_result.reasoning}")
                
                with tab4:
                    st.write("**Per-Skill Evidence Ledger**")
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Skill": skill.skill_name,
                                "Status": skill.status.value,
                                "Confidence": f"{skill.relevance_score:.0%}",
                                "Evidence Count": len(skill.evidence),
                                "Action Verbs": ", ".join(skill.evidence[0].action_verbs) if skill.evidence and skill.evidence[0].action_verbs else "N/A"
                            }
                            for skill in (report.validated_skills + report.weak_skills + report.ignored_skills)
                        ]),
                        use_container_width=True,
                        hide_index=True
                    )
                    
            except Exception as e:
                st.error(f"Error during semantic analysis: {e}")
                st.info("Please ensure both JD and Resume are properly formatted and contain skill-related content.")
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_technical_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Generating technical questions..."):
                ctx_jd = retrieve_context(vs, "jd", "technical stack, tools, methodologies, domain", k=k_retrieval, search_type=search_type)
                ctx_cv = retrieve_context(vs, "resume", "skills, tools, technologies, project details", k=k_retrieval, search_type=search_type)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_TECHNICAL_Q, context)
            st.subheader("Technical Questions")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_coding_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Generating coding questions..."):
                ctx_jd = retrieve_context(vs, "jd", "coding tasks, programming languages, data processing, testing", k=k_retrieval, search_type=search_type)
                ctx_cv = retrieve_context(vs, "resume", "coding experience, problems solved, libraries, pipelines, testing", k=k_retrieval, search_type=search_type)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_CODING_Q, context)
            st.subheader("Coding Questions")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_domain:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Running domain-fit analysis..."):
                ctx_jd = retrieve_context(vs, "jd", "domain, business context, analytics, industry", k=k_retrieval, search_type=search_type)
                ctx_cv = retrieve_context(vs, "resume", "domain experience, projects, industry exposure", k=k_retrieval, search_type=search_type)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_DOMAIN, context)
            st.subheader("Domain Expert Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_manager:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Running technical-fit analysis..."):
                ctx_jd = retrieve_context(vs, "jd", "required skills and years of experience, tooling, architecture", k=k_retrieval, search_type=search_type)
                ctx_cv = retrieve_context(vs, "resume", "skills with experience, projects, responsibilities", k=k_retrieval, search_type=search_type)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_MANAGER, context)
            st.subheader("Technical Manager Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_jd_summarization:
    if jd_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Summarizing JD..."):
                context = retrieve_context(vs, "jd", "summarize job description responsibilities skills qualifications", k=k_retrieval, search_type=search_type)
                answer = call_llm_with_context(PROMPT_JD_SUMMARY, context)
            st.subheader("Job Description Summary")
            st.write(answer)
    else:
        st.info("Please upload a Job Description to proceed.")

elif submit_jd_clarification:
    if jd_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Drafting clarification questions..."):
                context = retrieve_context(vs, "jd", "technical scope, tools, platforms, expectations, project details", k=k_retrieval, search_type=search_type)
                answer = call_llm_with_context(PROMPT_JD_CLARIFICATION, context)
            st.subheader("JD Clarification Questions")
            st.write(answer)
    else:
        st.info("Please upload a Job Description to proceed.")

elif submit_skill_analysis:
    if uploaded_resume is not None and top_skills:
        vs = ensure_vs()
        if vs:
            with st.spinner("Analyzing top skills in the resume..."):
                context = retrieve_context(vs, "resume", f"{top_skills}. roles, projects, responsibilities, dates, durations", k=k_retrieval, search_type=search_type)
                answer = call_llm_with_context(PROMPT_SKILL_ANALYST, context, top_skills=top_skills)
            st.subheader("Top Skill Analysis")
            st.write(answer)
    else:
        st.info("Please upload a Resume and enter Top Skills to proceed.")

elif submit_general_query:
    if jd_content or resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Answering your query..."):
                ctx_jd = retrieve_context(vs, "jd", input_promp or "requirements and skills", k=max(2, k_retrieval - 2), search_type=search_type) if jd_content else ""
                ctx_cv = retrieve_context(vs, "resume", input_promp or "candidate skills and projects", k=max(2, k_retrieval - 2), search_type=search_type) if resume_content else ""
                context = (ctx_jd + "\n\n---\n\n" + ctx_cv).strip()
                answer = call_llm_with_context(PROMPT_GENERAL_Q, context, user_query=input_promp or "Provide insights based on the context.")
            st.subheader("Query Response")
            st.write(answer)
    else:
        st.info("Please upload a Resume or a Job Description to proceed.")
