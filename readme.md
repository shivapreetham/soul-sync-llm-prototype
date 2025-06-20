# Soul Sync LLM Prototype

## Setup
1. Clone repo.
2. In project folder: `python3 -m venv venv` and activate.
3. `pip install -r requirements.txt`.

## Run
- `streamlit run light.py` to test generation with GPT-Neo 125M or the smaller models. The models will be downloaded at the first run and cached onto your system. Note that the loading time is always constant.
- `streamlit run heavy.py` to test generation with heavier models and more specific fall backs, works the same as the light model in terms of architecture.
- (Optional) Export to ONNX with `python -m transformers.onnx --model=EleutherAI/gpt-neo-125M onnx/` then `python generate_onnx.py`.

## Notes
- CPU-only: may be slow for larger models. Start with 125M.
