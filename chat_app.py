
import streamlit as st
import torch
from llm_utils import load_model, generate_response
from chat_utils import build_prompt, truncate_history

st.set_page_config(page_title="Soul Sync Chat", layout="centered")

@st.cache_resource
def init_model(model_name: str):
    return load_model(model_name)

def main():
    st.title("🤖 Soul Sync Chat")
    st.markdown("*An emotion-aware chatbot for meaningful conversations*")

    # Sidebar controls
    model_name = st.sidebar.selectbox(
        "Model",
        [
            "microsoft/DialoGPT-medium",  # Better than small, still under 2B
            "distilgpt2", 
            "gpt2"
        ], 
        index=0
    )
    
    tokenizer, model, device = init_model(model_name)

    # Better generation parameters to reduce repetition
    temperature = st.sidebar.slider("Temperature", 0.5, 1.2, 0.8)
    top_p = st.sidebar.slider("Top-p", 0.3, 0.95, 0.85)
    top_k = st.sidebar.slider("Top-k", 20, 100, 50)
    max_new_tokens = st.sidebar.number_input(
        "Max new tokens", min_value=15, max_value=100, value=40, step=5
    )

    emotion_label = st.sidebar.selectbox(
        "Detected Emotion",
        ["neutral", "happy", "excited", "sad", "frustrated", "confused", "angry"], 
        index=0
    )

    # Initialize chat history with better structure
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.conversation_context = ""

    # Clear chat button
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        st.session_state.conversation_context = ""

    # Display chat with better styling
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
            col1, col2 = st.columns([1, 4])
            with col2:
                st.markdown(f"🧑 **You:** {user_msg}")
                st.markdown(f"🤖 **Soul Sync:** {bot_msg}")
                if i < len(st.session_state.history) - 1:
                    st.markdown("---")

    # Input handling
    def submit():
        user_input = st.session_state.user_input.strip()
        if not user_input:
            return

        # Better history management
        safe_hist = truncate_history(
            st.session_state.history, tokenizer, max_tokens=300
        )
        
        # Build better prompt
        prompt = build_prompt(
            safe_hist, 
            user_input,
            emotion_label=emotion_label,
            model_type="dialogpt" if "DialoGPT" in model_name else "gpt",
            tokenizer=tokenizer
        )
        
        # Show debug info in sidebar
        with st.sidebar.expander("Debug Info"):
            st.code(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
            st.text_area("Generated prompt:", prompt, height=100)

        # Generate with better parameters
        with st.spinner("Thinking..."):
            bot_resp = generate_response(
                prompt,
                tokenizer,
                model,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                history=st.session_state.history  # Pass history for better context
            )
        
        # Update history
        st.session_state.history.append((user_input, bot_resp))
        st.session_state.user_input = ""
        st.rerun()

    # Better input interface
    st.markdown("### 💬 Chat with Soul Sync")
    st.text_input(
        "Type your message:", 
        key="user_input", 
        on_change=submit,
        placeholder="How are you feeling today?"
    )
    
    # Show some helpful prompts
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 I'm happy"):
            st.session_state.user_input = "I'm feeling really happy today!"
            submit()
    with col2:
        if st.button("😔 I'm sad"):
            st.session_state.user_input = "I'm feeling a bit down today"
            submit()
    with col3:
        if st.button("😕 I'm confused"):
            st.session_state.user_input = "I'm feeling confused about something"
            submit()

if __name__ == "__main__":
    main()
