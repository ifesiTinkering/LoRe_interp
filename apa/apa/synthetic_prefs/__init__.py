"""LoRe evaluation, dataset suitability diagnostics, and historical preference generation."""

from apa.synthetic_prefs.eval_prefs import evaluate_suitability, embed_preferences, report, load_prefs
from apa.synthetic_prefs.historical_prefs import (
    load_hist_llama,
    generate_historical_preferences,
    preference_from_logprobs,
    preferences_to_labels,
)
