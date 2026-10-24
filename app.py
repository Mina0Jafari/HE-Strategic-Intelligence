
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from document_retriever import search_documents
from agents.orchestrator_agent import OrchestratorAgent
# -----------------------------------
# PROJECT PATHS
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"
CSS_FILE = BASE_DIR / "styles" / "style.css"


# -----------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------
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
with open(CSS_FILE, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

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
    st.caption("Developed by Mina Jafari")

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
        st.error("OpenAI API key not found. Please check your .env file.")

    else:
        try:
            with st.spinner("Running multi-agent strategic analysis..."):

                orchestrator = OrchestratorAgent(
                    top_k=3,
                    model="gpt-4.1-mini"
                )

                result = orchestrator.run(question)

            if result["status"] == "completed":

                st.success(
                    "Multi-agent strategic analysis completed!"
                )

                st.markdown(result["report"])

                st.divider()

                # AGENT EXECUTION DETAILS
                st.subheader("Agent Execution")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Execution Time",
                        f"{result['total_seconds']} seconds"
                    )

                with col2:
                    st.metric(
                        "Evidence Sources",
                        len(result["evidence"])
                    )

                for entry in result["execution_log"]:
                    st.write(
                        f"✓ {entry['agent']}: {entry['status']}"
                    )

                # RETRIEVED EVIDENCE
                st.divider()
                st.subheader("Supporting Evidence")

                for item in result["evidence"]:
                    with st.expander(
                        f"{item['source_id']}: {item['section']}"
                    ):
                        st.write(
                            f"**Document:** {item['document']}"
                        )
                        st.write(
                            f"**Section:** {item['section']}"
                        )
                        st.code(
                            item["content"],
                            language=None
                        )

            else:
                st.warning(result["report"])

        except Exception as e:
            st.error("Unable to complete multi-agent analysis.")
            st.exception(e)



# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()

st.caption(
    "Developed by Mina Jafari | MSc Data Science "
)
