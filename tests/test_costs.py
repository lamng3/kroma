"""KROMA prices a run from MetricsListener's token counts."""

from kroma.costs import api_cost


def test_api_cost_prices_known_models_and_leaves_local_ones_unpriced(monkeypatch):
    api = {"input_token": 400_000, "output_token": 80_000, "token_count": 480_000, "api_call_cnt": 1000}
    cost = api_cost("gpt-4o-mini", api, pairs=1000)
    assert cost["cost_usd"] == round((400_000 * 0.15 + 80_000 * 0.60) / 1e6, 6)
    assert cost["tokens_per_call"] == 480 and cost["cost_usd_per_pair"] == round(cost["cost_usd"] / 1000, 6)
    assert api_cost("meta-llama/Llama-3.3-70B-Instruct", api)["cost_usd"] is None
    monkeypatch.setenv("KROMA_PRICES", '{"llama-3.3-70b-instruct": [0.6, 0.6]}')
    assert api_cost("meta-llama/Llama-3.3-70B-Instruct", api)["cost_usd"] == 0.288
