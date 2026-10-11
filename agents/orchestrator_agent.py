
import time

from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from agents.reporting_agent import ReportingAgent


class OrchestratorAgent:
    """
    Coordinates retrieval, analysis and reporting
    for strategic decision support.
    """

    def __init__(self, top_k=3, model="gpt-4.1-mini"):
        self.name = "Orchestrator Agent"

        self.retrieval_agent = RetrievalAgent(top_k=top_k)
        self.analysis_agent = AnalysisAgent(model=model)
        self.reporting_agent = ReportingAgent(model=model)

    def run(self, question):
        if not question.strip():
            raise ValueError("A strategic question is required.")

        start_time = time.perf_counter()
        execution_log = []

        # STEP 1: RETRIEVAL
        retrieval_result = self.retrieval_agent.run(question)

        evidence = retrieval_result["evidence"]

        execution_log.append({
            "agent": "Retrieval Agent",
            "status": "completed",
            "evidence_count": len(evidence)
        })

        # Stop if no evidence was found
        if not evidence:
            return {
                "question": question,
                "status": "insufficient_evidence",
                "report": "No supporting evidence was retrieved.",
                "evidence": [],
                "execution_log": execution_log,
                "total_seconds": round(
                    time.perf_counter() - start_time, 2
                )
            }

        # STEP 2: ANALYSIS
        analysis_result = self.analysis_agent.run(
            question=question,
            evidence=evidence
        )

        execution_log.append({
            "agent": "Analysis & Reasoning Agent",
            "status": "completed"
        })

        # STEP 3: REPORTING
        reporting_result = self.reporting_agent.run(
            question=question,
            analysis=analysis_result["analysis"],
            evidence=evidence
        )

        execution_log.append({
            "agent": "Reporting Agent",
            "status": "completed"
        })

        # FINAL RESULT
        return {
            "question": question,
            "status": "completed",
            "report": reporting_result["report"],
            "analysis": analysis_result["analysis"],
            "evidence": evidence,
            "execution_log": execution_log,
            "total_seconds": round(
                time.perf_counter() - start_time, 2
            )
        }


# -----------------------------------
# TEST THE COMPLETE MULTI-AGENT SYSTEM
# -----------------------------------
if __name__ == "__main__":
    question = (
        "What are the main student retention "
        "challenges and strategic priorities?"
    )

    orchestrator = OrchestratorAgent(
        top_k=3,
        model="gpt-4.1-mini"
    )

    result = orchestrator.run(question)

    print("\n" + "=" * 60)
    print("HE STRATEGIC INTELLIGENCE")
    print("FOUR-AGENT WORKFLOW")
    print("=" * 60)

    print(f"\nStatus: {result['status']}")
    print(f"Execution time: {result['total_seconds']} seconds")

    print("\nAGENT EXECUTION LOG")
    for entry in result["execution_log"]:
        print(f"- {entry['agent']}: {entry['status']}")

    print("\nSTRATEGIC REPORT")
    print(result["report"])
