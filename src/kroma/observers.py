"""In-process listener for token counts and refinement timings."""


class MetricsListener:
    """Collects API counters and refinement timings for one run.

    This is a local observer for the matching loop. It does not record
    conversation traces.
    """

    def __init__(self):
        self.api = {
            "input_token": 0,
            "output_token": 0,
            "token_count": 0,
            "api_call_cnt": 0,
        }
        self.timings = {
            "offline_refinement": 0.0,
            "online_refinement": 0.0,
        }

    def add_api(self, metrics: dict) -> None:
        for key in self.api:
            self.api[key] += metrics.get(key, 0)

    def add_timing(self, bucket: str, seconds: float) -> None:
        self.timings[bucket] = self.timings.get(bucket, 0.0) + seconds
