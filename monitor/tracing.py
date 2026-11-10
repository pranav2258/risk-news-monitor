"""Lightweight tracing: record every node run (latency, outcome, token estimate) to JSONL.

A stand-in for an observability platform such as Galileo or LangSmith, which captures
the same things (plus full inputs and outputs) and adds dashboards and quality metrics.
"""

import json
import time
from pathlib import Path

try:
    # interrupt() raises this to pause the graph for a human. It is not a failure.
    from langgraph.errors import GraphBubbleUp
except ImportError:  # lets the non-graph modules be used without langgraph installed
    GraphBubbleUp = ()


class Tracer:
    def __init__(self, path: str | Path | None = None, reset: bool = True):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path and reset:
            self.path.write_text("", encoding="utf-8")

    def wrap(self, node_name: str, fn):
        """Return a version of `fn` that records one trace event per call."""
        def traced(state):
            start = time.perf_counter()
            status, error, output = "ok", None, None
            try:
                output = fn(state)
                return output
            except GraphBubbleUp:
                status = "paused"
                raise
            except Exception as e:
                status, error = "error", f"{type(e).__name__}: {e}"
                raise
            finally:
                self._record({
                    "run_id": state.get("article_id"),
                    "node": node_name,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    "prompt_tokens": output.get("prompt_tokens") if isinstance(output, dict) else None,
                    "error": error,
                })

        traced.__name__ = node_name
        return traced

    def _record(self, event: dict) -> None:
        self.events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def summary(self) -> dict:
        nodes: dict[str, dict] = {}
        for e in self.events:
            n = nodes.setdefault(e["node"], {"calls": 0, "total_ms": 0.0, "errors": 0, "paused": 0})
            n["calls"] += 1
            n["total_ms"] += e["latency_ms"]
            n["errors"] += e["status"] == "error"
            n["paused"] += e["status"] == "paused"
        tokens = sum(e["prompt_tokens"] or 0 for e in self.events)
        return {"nodes": nodes, "prompt_tokens": tokens, "runs": len({e["run_id"] for e in self.events})}

    def format_summary(self) -> str:
        s = self.summary()
        lines = [f"{'node':<20}{'calls':>6}{'avg ms':>9}{'errors':>8}{'paused':>8}"]
        for name, n in s["nodes"].items():
            lines.append(f"{name:<20}{n['calls']:>6}{n['total_ms'] / n['calls']:>9.2f}"
                         f"{n['errors']:>8}{n['paused']:>8}")
        lines.append(f"runs: {s['runs']}   estimated prompt tokens: {s['prompt_tokens']}"
                     "   (cost = tokens x your model's price per token)")
        return "\n".join(lines)
