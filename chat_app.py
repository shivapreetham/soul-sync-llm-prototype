# chat_app.py
import streamlit as st
import torch
from llm_utils import load_tokenizer_and_model, generate_response
from chat_utils import build_chat_prompt, truncate_history

st.set_page_config(page_title="Soul Sync Chat", layout="centered")

@st.cache_resource
def init_model(model_name):
    device = torch.device("cpu")
    tokenizer, model, device = load_tokenizer_and_model(model_name, device)
    return tokenizer, model, device

def main():
    st.title("Soul Sync Base Chat")

    # Sidebar settings
    model_name = st.sidebar.selectbox("Model", ["distilgpt2", "EleutherAI/gpt-neo-125M"])
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    top_p = st.sidebar.slider("Top-p (nucleus)", 0.1, 1.0, 0.9)
    max_length = st.sidebar.number_input("Max new tokens", min_value=10, max_value=200, value=50, step=10)

    tokenizer, model, device = init_model(model_name)

    # Initialize history in session_state
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (user_msg, bot_msg)

    # Display past chat
    for user_msg, bot_msg in st.session_state.history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Bot:** {bot_msg}")

    # Define callback for when user submits input
    def submit():
        user_input = st.session_state.input_text.strip()
        if not user_input:
            return
        # Truncate history to fit context window
        st.session_state.history = truncate_history(st.session_state.history, tokenizer, max_tokens=512)

        # Build prompt
        prompt = build_chat_prompt(st.session_state.history, user_input)

        # Generate response
        with st.spinner("Generating..."):
            full_text, elapsed = generate_response(
                prompt, tokenizer, model, device,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                warmup=True
            )
        # Strip the prompt from the generated text if it appears at start
        # This assumes generate_response returns full prompt+continuation or just continuation.
        # Often HF generate returns only continuation, but if full prompt appears, remove it:
        if full_text.startswith(prompt):
            bot_response = full_text[len(prompt):].strip()
        else:
            bot_response = full_text.strip()

        # Append to history
        st.session_state.history.append((user_input, bot_response))
        # Clear input
        st.session_state.input_text = ""  # since clear_on_submit not available, we clear here

    # Text input with callback
    st.text_input("Your message:", key="input_text", on_change=submit)

if __name__ == "__main__":
    main()
