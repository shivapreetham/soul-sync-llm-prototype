# Soul Sync LLM Prototype

## Setup
1. Clone repo.
2. In project folder: `python3 -m venv venv` and activate.
3. `pip install -r requirements.txt`.

## Run
- `streamlit run .py` to test generation with GPT-Neo 125M.
- (Optional) Export to ONNX with `python -m transformers.onnx --model=EleutherAI/gpt-neo-125M onnx/` then `python generate_onnx.py`.

## Notes
- CPU-only: may be slow for larger models. Start with 125M.
