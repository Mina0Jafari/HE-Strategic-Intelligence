"""Offline regression checks; run from project root: python smoke_test_backend.py"""
from unittest.mock import patch
import pandas as pd
from agents.retrieval_agent import RetrievalAgent

rows = pd.DataFrame({"School": ["Education", "Health"], "2023/24": [90, 89], "2024/25": [87, 86], "Change (pp)": [-3, -3]})
doc = {"source": "uel_annual_report_2024_25.pdf", "section": "Page 3",
       "text": "Vision 2028", "distance": 0.2, "source_type": "pdf", "data_type": "public"}
with (patch('agents.retrieval_agent.search_documents', return_value=[doc]),
      patch('agents.retrieval_agent.analyse_retention', return_value=rows)):
    agent = RetrievalAgent(top_k=3)
    result = agent.run('How does undergraduate retention vary by school?')
    evidence = result['evidence']
    assert len(evidence) == 2
    assert evidence[0]['data_type'] == 'public'
    assert evidence[1]['data_type'] == 'simulated'
    assert 'Education' in evidence[1]['content']
    assert '-3' in evidence[1]['content']
    assert result['evidence_count'] == 2
print('PASS: public provenance and deterministic school comparison are retained.')
