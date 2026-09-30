from dataclasses import dataclass, fields

# global token limit constant (adjust as needed)

@dataclass
class APIStats:
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0