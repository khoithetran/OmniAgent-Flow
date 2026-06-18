"""Verify context window helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_gradio import (
    MODELS, DEFAULT_MODEL, CONTEXT_WINDOWS,
    _format_context_window, _format_usage, _button_variants,
    _estimate_tokens, build_ui
)

print("Format usage:")
for used, total in [(0, 128000), (500, 128000), (3200, 128000),
                     (25000, 128000), (200000, 200000), (1500000, 1500000)]:
    print("  used=%8d total=%8d -> %s" % (used, total, _format_usage(used, total)))

print()
print("Estimate tokens:")
for txt in ["", "Hello", "Xin chào, tôi là trợ lý ảo", "A" * 4000]:
    print("  len=%-5d -> %d tokens" % (len(txt), _estimate_tokens(txt)))

print()
print("Button variants for selected=gpt-4o:")
variants = _button_variants("gpt-4o")
for m, v in zip(MODELS, variants):
    print("  %-20s variant=%s" % (m, v.get("variant", "?")))

print()
print("Button variants for selected=gpt-4o-mini (default):")
variants = _button_variants("gpt-4o-mini")
for m, v in zip(MODELS, variants):
    print("  %-20s variant=%s" % (m, v.get("variant", "?")))

print()
demo = build_ui()
print("UI built OK")
