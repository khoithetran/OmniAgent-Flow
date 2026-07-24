"""Verify model button click handler returns correct number of values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app_gradio as ag
import gradio as gr

with gr.Blocks() as demo:
    state = gr.State({"selected_model": ag.DEFAULT_MODEL, "kb_ready": False,
                       "kb_domain": "", "kb_pages": 0, "kb_chunks": 0, "kb_url": ""})
    model_buttons = ag._build_model_buttons()

    def _make_select_handler(m):
        def _handler(s):
            new_state = ag.select_model(s, m)[0]
            variants = ag._button_variants(m)
            label = ag._context_window_label(m)
            return (new_state, *variants, label)
        return _handler

    # Test handler for OpenAI - GPT 5.4
    handler = _make_select_handler("OpenAI - GPT 5.4")
    s = {"selected_model": ag.DEFAULT_MODEL, "kb_ready": False,
         "kb_domain": "", "kb_pages": 0, "kb_chunks": 0, "kb_url": ""}
    result = handler(s)
    print("Number of returned values:", len(result))
    print("Value 0 (state):", result[0])
    print("Value 1 (btn1 update):", result[1])
    print("Value 2 (btn2 update):", result[2])
    print("Value 3 (label):", result[3])
    assert len(result) == 4
