from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src import llm_assistant


ROOT = Path(__file__).resolve().parents[1]


def test_llm_ranking_intent_reaches_deterministic_query(monkeypatch):
    captured = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=llm_assistant.PortfolioQuerySpec(
                    answerable=True,
                    decline_reason=None,
                    metric="default_rate",
                    group_by="loan_type",
                    filters=[],
                    ranking_intent="lowest",
                    top_n=5,
                )
            )

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(llm_assistant, "OpenAI", FakeOpenAI)
    account = pd.read_csv(ROOT / "data" / "processed" / "account_scored.csv")

    result = llm_assistant.ask_portfolio(
        "Which loan type defaults the least?",
        account,
        api_key="test-key",
    )

    assert captured["input"] == "Which loan type defaults the least?"
    assert captured["text_format"] is llm_assistant.PortfolioQuerySpec
    assert result.table.iloc[0]["loan_type"] == "Auto Loan"
    assert "lowest default rate at 13.9%" in result.answer
    assert "ranking: lowest" in result.provenance
