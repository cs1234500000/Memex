"""
memex/config.py
Central model configuration.

Two tiers:
  EXPERT_MODEL      — high-capability model for reasoning, synthesis, and analysis.
                      Used by: all forum agents, The Debate, decomposer,
                               report agent, summarizer, knowledge tagger.

  LIGHTWEIGHT_MODEL — fast, cheap model for high-volume binary/classification tasks.
                      Used by: the ingest filter's per-item classification step.

Override either via environment variables.
"""

import os

EXPERT_MODEL: str = os.environ.get("EXPERT_MODEL", "gpt-4o")
LIGHTWEIGHT_MODEL: str = os.environ.get("LIGHTWEIGHT_MODEL", "gpt-4o-mini")
