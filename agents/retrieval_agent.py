
from retrieval.document_retriever import search_documents


class RetrievalAgent:
    """
    Retrieves relevant evidence from the institutional
    knowledge base for strategic decision support.
    """

    def __init__(self, top_k=3):
        self.top_k = top_k
        self.name = "Retrieval Agent"

    def run(self, question):
        if not question.strip():
            raise ValueError("A strategic question is required.")

        documents = search_documents(
            question=question,
            n_results=self.top_k
        )

        evidence = []

        for index, doc in enumerate(documents, start=1):
            evidence.append({
                "source_id": f"Source {index}",
                "document": doc["source"],
                "section": doc["section"],
                "content": doc["text"],
                "distance": doc["distance"]
            })

        return {
            "agent": self.name,
            "question": question,
            "evidence": evidence,
            "evidence_count": len(evidence)
        }


if __name__ == "__main__":
    agent = RetrievalAgent(top_k=3)

    result = agent.run(
        "What are the main student retention challenges?"
    )

    print(f"\nAgent: {result['agent']}")
    print(f"Evidence found: {result['evidence_count']}")

    for item in result["evidence"]:
        print(f"\n{item['source_id']}")
        print(f"Section: {item['section']}")
        print(f"Document: {item['document']}")
        print(item["content"])
