"""What a KROMA run costs: dollars at list price from the token counts MetricsListener keeps.

`MetricsListener.api` counts input and output tokens and API calls for a run.
`api_cost` prices them for the run's model. Models served locally (Ollama,
vLLM, Together endpoints you host) have no list price here; their cost is the
run's time, already in `timings`. Extend or override prices with
KROMA_PRICES, a JSON object of dollars per million tokens, `{"model": [input, output]}`.
"""

import json
import os

PRICES = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def prices():
    table = dict(PRICES)
    table.update({k: tuple(v) for k, v in json.loads(os.environ.get("KROMA_PRICES", "{}")).items()})
    return table


def api_cost(model, api, pairs=0):
    """Dollars (None if the model has no price) and per-call and per-pair figures for one run."""
    tokens_in, tokens_out = api.get("input_token", 0), api.get("output_token", 0)
    calls = api.get("api_call_cnt", 0)
    rate = prices().get(str(model).split("/")[-1].lower())
    usd = None if rate is None else (tokens_in * rate[0] + tokens_out * rate[1]) / 1e6
    return {
        "model": model,
        "cost_usd": None if usd is None else round(usd, 6),
        "api_calls": calls,
        "tokens_per_call": round((tokens_in + tokens_out) / calls) if calls else 0,
        "cost_usd_per_pair": None if usd is None or not pairs else round(usd / pairs, 6),
    }
