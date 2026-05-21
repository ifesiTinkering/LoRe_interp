"""
Response generation strategies for democratic inference.
"""

from __future__ import annotations

from typing import Any

import torch


def temperature_sampling(
    model: Any,
    tokenizer: Any,
    query: str,
    k: int,
    config: dict,
) -> list[str]:
    """Generate k responses using temperature sampling for diversity."""
    temperature = config.get('temperature', 1.2)
    max_new_tokens = config.get('max_new_tokens', 512)

    messages = [{"role": "user", "content": query}]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"User: {query}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(k):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        responses.append(response)

    return responses
