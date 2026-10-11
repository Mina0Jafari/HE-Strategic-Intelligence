import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Locate .env in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Load API credentials
load_dotenv(dotenv_path=ENV_FILE, override=True)


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="HE Strategic Intelligence | Mina Jafari",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# CUSTOM STYLING
# -----------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    h1, h2, h3 {
        color: #003054;
    }

    .project-banner {
        background-color: #003054;
        padding: 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }

    .project-banner h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 10px;
    }

    .project-banner p {
        color: #E0EAF2;
        font-size: 16px;
        margin-bottom: 0;
    }

    .info-card {
        background-color: white;
        padding: 20px;
        border: 1px solid #E0E6ED;
        border-radius: 10px;
        min-height: 115px;
    }

    .info-card h4 {
        color: #003054;
        margin: 0 0 8px 0;
    }

    .info-card p {
        color: #526575;
        margin: 0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #003054;
        border-color: #003054;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.title("HE Strategic Intelligence")
    st.caption("MSc Data Science Dissertation")

    st.divider()

    st.subheader("Project Information")
    st.write("**Researcher:** Mina Jafari")
    st.write("**Supervisor:** Dr. Sujit Biswas")
    st.write("**Programme:** MSc Data Science")

    st.divider()

    st.subheader("System Architecture")
    st.write("1. Orchestrator Agent")
    st.write("2. Retrieval Agent")
    st.write("3. Analysis & Reasoning Agent")
    st.write("4. Reporting Agent")

    st.divider()
    st.caption("Prototype | Development Stage")

# -----------------------------------
# MAIN PAGE
# -----------------------------------
st.markdown("""
<div class="project-banner">
    <h1>HE Strategic Intelligence</h1>
    <p>Multi-Agent AI for Evidence-Based Strategic Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.caption("MSc DATA SCIENCE | INDIVIDUAL RESEARCH PROJECT")

st.header("Project Overview")

st.write("""
**A Multi-Agent AI Framework for Strategic Decision Support
in Higher Education: Design, Implementation, and Evaluation**
""")

st.write("""
This research project explores how a multi-agent artificial
intelligence system can support strategic decision-making in
higher education by retrieving, analysing and synthesising
information from institutional and sector-wide data sources.
""")

# -----------------------------------
# PROJECT COMPONENTS
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>Data Retrieval</h4>
        <p>Retrieve relevant institutional and sector evidence.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>AI Analysis</h4>
        <p>Interpret evidence to support strategic decisions.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>Strategic Reporting</h4>
        <p>Generate evidence-based findings and recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------
# STRATEGIC QUESTION INTERFACE
# -----------------------------------
st.header("Strategic Analysis Workspace")

st.write("""
Enter a strategic question relating to higher education.
The completed system will use specialised AI agents to
retrieve evidence, analyse findings and generate recommendations.
""")

question = st.text_area(
    "Your Strategic Question",
    placeholder=(
        "How can universities improve student retention "
        "and student satisfaction?"
    ),
    height=120
)


if st.button("Analyse Question", type="primary"):

    if not question.strip():
        st.warning("Please enter a strategic question.")

    elif not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OpenAI API key not found. "
            "Please check your .env file."
        )

    else:
        try:
            client = OpenAI()

            with st.spinner("Generating strategic analysis..."):

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    instructions="""
                    You are an AI strategic decision-support
                    assistant specialising in higher education.

                    Your role is to support evidence-based
                    organisational strategy and planning.

                    Structure your response using:

                    1. Strategic Overview
                    2. Key Considerations
                    3. Strategic Opportunities
                    4. Risks and Challenges
                    5. Recommended Actions

                    Use clear, professional language.

                    Distinguish facts from assumptions.
                    Do not invent statistics, institutional
                    evidence, or references.

                    If institutional data has not been
                    provided, explain that recommendations
                    are preliminary and require validation.
                    """,
                    input=question
                )

            st.success("Strategic analysis completed!")

            st.subheader("AI Strategic Analysis")

            st.markdown(response.output_text)

        except Exception as e:
            st.error(
                "Unable to generate analysis. "
                "Please check your API configuration "
                "and billing."
            )
            st.exception(e)

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()

st.caption(
    "Developed by Mina Jafari | MSc Data Science "
)
