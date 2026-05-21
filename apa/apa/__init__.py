"""
APA: Aggregated Preference Alignment

A pipeline for democratic preference aggregation that:
1. Learns individual user reward models using LoRe on PRISM
2. Simulates "future" users via ProgressGym HistLlama models
3. Generates diverse response slates and aggregates preferences democratically
"""

from apa.config import APAConfig, get_config, configure_environment

__version__ = "0.1.0"
__all__ = [
    "APAConfig",
    "get_config",
    "configure_environment",
]
