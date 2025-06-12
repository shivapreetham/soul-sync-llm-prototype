import streamlit as st
import torch
from context.context_aware import ContextualChatSystem
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Soul Sync - Intelligent Chat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_chat_system(model_name: str):
    """Initialize the contextual chat system"""
    return ContextualChatSystem(model_name)

def plot_context_relevance(insights):
    """Plot context relevance over time"""
    if not insights.get('recent_relevance'):
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=insights['recent_relevance'],
        mode='lines+markers',
        name='Context Relevance',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                  annotation_text="Context Threshold")
    
    fig.update_layout(
        title="Context Relevance Over Recent Exchanges",
        xaxis_title="Recent Exchanges",
        yaxis_title="Relevance Score",
        height=300,
        showlegend=False
    )
    
    return fig

def main():
    st.title("🧠 Soul Sync - Intelligent Context-Aware Chat")
    st.markdown("*An AI that understands conversation context and emotional nuance*")

    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        model_name = st.selectbox(
            "Choose Model",
            [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small",
                "distilgpt2",
                "gpt2"
            ],
            index=0,
            help="DialoGPT models are better for conversations, GPT-2 for general knowledge"
        )
        
        # Initialize chat system
        try:
            with st.spinner("Loading AI model..."):
                chat_system = init_chat_system(model_name)
            st.success(f"✅ {model_name} loaded on {chat_system.device}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.stop()
        
        st.divider()
        
        # Generation Parameters
        st.header("⚙️ Generation Settings")
        temperature = st.slider(
            "Temperature", 
            0.5, 1.2, 0.7, 0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        max_tokens = st.slider(
            "Max Response Length", 
            15, 60, 30, 5,
            help="Maximum tokens in response"
        )
        
        context_threshold = st.slider(
            "Context Sensitivity",
            0.1, 0.8, 0.3, 0.1,
            help="How much context similarity needed to use history"
        )
        
        # Update system threshold
        chat_system.similarity_threshold = context_threshold
        
        st.divider()
        
        # Emotion Context
        st.header("😊 Emotional Context")
        emotion_label = st.selectbox(
            "Detected Emotion",
            ["neutral", "happy", "excited", "sad", "frustrated", "confused", "angry"],
            help="This affects how the AI responds to you"
        )
        
        st.divider()
        
        # Debug Section
        debug_mode = st.checkbox("🔍 Debug Mode", help="Show AI reasoning process")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.history = []
            chat_system.conversation_memory = []
            st.rerun()

    # Main Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        # Initialize chat history
        if "history" not in st.session_state:
            st.session_state.history = []
        
        # Display chat
        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.history:
                st.info("👋 Start a conversation! I can understand context and respond emotionally.")
            
            for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
                with st.chat_message("user"):
                    st.write(user_msg)
                
                with st.chat_message("assistant"):
                    st.write(bot_msg)
        
        # Input Interface
        def handle_input():
            user_input = st.session_state.user_input.strip()
            if not user_input:
                return
            
            with st.spinner("🤖 AI is thinking contextually..."):
                # Generate response using intelligent system
                response = chat_system.generate_response(
                    user_input=user_input,
                    history=st.session_state.history,
                    emotion_label=emotion_label,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                
                # Handle empty responses
                if not response:
                    response = chat_system._get_contextual_fallback(
                        user_input, 
                        chat_system.determine_conversation_mode(user_input, st.session_state.history),
                        emotion_label
                    )
            
            # Update history
            st.session_state.history.append((user_input, response))
            st.session_state.user_input = ""
            st.rerun()
        
        st.text_input(
            "💭 Type your message:",
            key="user_input",
            on_change=handle_input,
            placeholder="Try: 'Hi there!' or 'What's the capital of France?' or 'I'm feeling sad today'"
        )
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            if st.button("👋 Greet"):
                st.session_state.user_input = "Hello! How are you doing?"
                handle_input()
        
        with col_b:
            if st.button("❓ Ask Question"):
                st.session_state.user_input = "What do you think about artificial intelligence?"
                handle_input()
        
        with col_c:
            if st.button("💭 Share Feeling"):
                st.session_state.user_input = "I've been thinking a lot about my future lately"
                handle_input()
        
        with col_d:
            if st.button("🔄 Follow Up"):
                if st.session_state.history:
                    st.session_state.user_input = "Can you tell me more about that?"
                    handle_input()
    
    # Analytics Panel
    with col2:
        st.header("📊 Conversation Analytics")
        
        if st.session_state.history:
            # Get conversation insights
            insights = chat_system.get_conversation_insights(st.session_state.history)
            
            # Display metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Exchanges", insights['total_exchanges'])
            with col_m2:
                st.metric("Avg Context Score", f"{insights['avg_relevance']:.2f}")
            
            st.metric("High Context Exchanges", 
                     f"{insights['high_context_exchanges']}/{insights['total_exchanges']}")
            
            # Plot context relevance
            if len(st.session_state.history) > 1:
                fig = plot_context_relevance(insights)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Current conversation mode analysis
            if st.session_state.history:
                last_input = st.session_state.history[-1][0]
                mode = chat_system.determine_conversation_mode(last_input, st.session_state.history[:-1])
                
                st.subheader("🧠 Last Input Analysis")
                st.write(f"**Context Relevance:** {mode['relevance_score']:.2f}")
                st.write(f"**Using Context:** {'✅' if mode['use_context'] else '❌'}")
                st.write(f"**Type:** {', '.join([k.replace('is_', '') for k, v in mode.items() if k.startswith('is_') and v])}")
        
        else:
            st.info("Start chatting to see analytics!")
        
        # Debug Information
        if debug_mode and st.session_state.history:
            st.subheader("🔍 Debug Info")
            
            last_input = st.session_state.history[-1][0]
            mode = chat_system.determine_conversation_mode(last_input, st.session_state.history[:-1])
            
            with st.expander("Conversation Mode Analysis"):
                st.json(mode)
            
            if len(st.session_state.history) > 1:
                prompt = chat_system.build_intelligent_prompt(
                    last_input, 
                    st.session_state.history[:-1], 
                    emotion_label
                )
                
                with st.expander("Generated Prompt"):
                    st.code(prompt)
                    st.write(f"Prompt length: {len(chat_system.tokenizer.encode(prompt))} tokens")

    # Help Section
    with st.expander("💡 How This AI Works"):
        st.markdown("""
        **Context Intelligence:**
        - Analyzes how your current message relates to previous conversation
        - Uses TF-IDF similarity and keyword matching
        - Detects pronouns and references that indicate context continuation
        
        **Conversation Modes:**
        - **High Context:** Heavily weights previous conversation
        - **Low Context:** Treats input as new topic
        - **Question Mode:** Focuses on providing clear answers
        - **Emotional Mode:** Prioritizes empathetic responses
        
        **Smart Features:**
        - Automatic context relevance detection
        - Emotional state awareness
        - Adaptive response generation
        - Memory management to prevent confusion
        
        **Tips for Better Conversations:**
        - Reference previous topics with "it", "that", "this"
        - Ask follow-up questions
        - Share your feelings and thoughts
        - Be specific about what you want to discuss
        """)

if __name__ == "__main__":
    main()