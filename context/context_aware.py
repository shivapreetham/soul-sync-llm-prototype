import torch
import re
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ContextualChatSystem:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer, self.model, self.device = self.load_model()
        self.conversation_memory = []
        self.context_window = 3  # Number of previous exchanges to consider
        self.similarity_threshold = 0.3  # Threshold for context relevance
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.stop_words = set(stopwords.words('english'))
        
    def load_model(self):
        """Load model with proper configuration"""
        print(f"Loading {self.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Proper tokenizer setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            model.to(device)
        
        model.eval()
        return tokenizer, model, device
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        try:
            tokens = word_tokenize(text.lower())
            keywords = [word for word in tokens 
                       if word.isalpha() and 
                       word not in self.stop_words and 
                       len(word) > 2]
            return keywords
        except:
            return text.lower().split()
    
    def calculate_context_relevance(self, current_input: str, history: List[Tuple[str, str]]) -> float:
        """
        Calculate how relevant current input is to recent conversation context
        Returns score between 0 and 1
        """
        if not history:
            return 0.0
        
        # Get recent context (last few exchanges)
        recent_context = " ".join([
            f"{user_msg} {bot_msg}" 
            for user_msg, bot_msg in history[-self.context_window:]
        ])
        
        if not recent_context.strip():
            return 0.0
        
        try:
            # Use TF-IDF similarity
            documents = [current_input, recent_context]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Also check for pronoun references (they, it, that, this)
            pronouns = ['it', 'they', 'that', 'this', 'he', 'she', 'them']
            current_words = current_input.lower().split()
            pronoun_bonus = 0.3 if any(pronoun in current_words for pronoun in pronouns) else 0.0
            
            # Check for question words that might relate to context
            question_words = ['what', 'why', 'how', 'when', 'where', 'who']
            question_bonus = 0.2 if any(qw in current_words for qw in question_words) else 0.0
            
            final_score = min(1.0, similarity + pronoun_bonus + question_bonus)
            return final_score
            
        except:
            # Fallback to keyword matching
            current_keywords = set(self.extract_keywords(current_input))
            context_keywords = set(self.extract_keywords(recent_context))
            
            if not current_keywords or not context_keywords:
                return 0.0
            
            overlap = len(current_keywords.intersection(context_keywords))
            union = len(current_keywords.union(context_keywords))
            
            return overlap / union if union > 0 else 0.0
    
    def determine_conversation_mode(self, user_input: str, history: List[Tuple[str, str]]) -> Dict:
        """
        Determine how to handle the conversation based on context relevance
        """
        relevance_score = self.calculate_context_relevance(user_input, history)
        
        # Classify input type
        input_lower = user_input.lower().strip()
        
        # Check if it's a greeting
        is_greeting = any(greeting in input_lower 
                         for greeting in ['hi', 'hello', 'hey', 'good morning', 'good evening'])
        
        # Check if it's a direct question
        is_question = (input_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')) or 
                      input_lower.endswith('?'))
        
        # Check if it's an emotional statement
        emotion_words = ['feel', 'feeling', 'sad', 'happy', 'angry', 'frustrated', 'excited', 'worried']
        is_emotional = any(word in input_lower for word in emotion_words)
        
        mode = {
            'relevance_score': relevance_score,
            'use_context': relevance_score > self.similarity_threshold,
            'is_greeting': is_greeting,
            'is_question': is_question,
            'is_emotional': is_emotional,
            'context_weight': min(0.8, relevance_score * 2),  # Weight for context importance
        }
        
        return mode
    
    def build_intelligent_prompt(self, user_input: str, history: List[Tuple[str, str]], 
                                emotion_label: str = "neutral") -> str:
        """
        Build context-aware prompts based on conversation analysis
        """
        mode = self.determine_conversation_mode(user_input, history)
        
        if "DialoGPT" in self.model_name:
            return self._build_dialogpt_prompt(user_input, history, mode, emotion_label)
        else:
            return self._build_gpt_prompt(user_input, history, mode, emotion_label)
    
    def _build_dialogpt_prompt(self, user_input: str, history: List[Tuple[str, str]], 
                              mode: Dict, emotion_label: str) -> str:
        """
        Build DialoGPT prompt with intelligent context handling
        """
        # For DialoGPT, we need to be very careful about format
        # It expects: input<EOS>response<EOS>input<EOS>response<EOS>...
        
        prompt_parts = []
        
        if mode['use_context'] and history:
            # Use weighted context - more recent exchanges get more weight
            context_exchanges = history[-2:] if mode['context_weight'] > 0.5 else history[-1:]
            
            for user_msg, bot_msg in context_exchanges:
                # Clean and format each exchange
                clean_user = user_msg.strip()
                clean_bot = bot_msg.strip()
                prompt_parts.append(f"{clean_user}{self.tokenizer.eos_token}{clean_bot}{self.tokenizer.eos_token}")
        
        # Add current input
        prompt_parts.append(f"{user_input.strip()}{self.tokenizer.eos_token}")
        
        return "".join(prompt_parts)
    
    def _build_gpt_prompt(self, user_input: str, history: List[Tuple[str, str]], 
                         mode: Dict, emotion_label: str) -> str:
        """
        Build GPT-2 style prompt with context intelligence
        """
        lines = []
        
        # System prompt - make it conversational, not instructional
        if mode['is_emotional']:
            lines.append("This is a conversation with an empathetic AI assistant.")
        elif mode['is_question']:
            lines.append("This is a conversation with a helpful AI assistant.")
        else:
            lines.append("This is a friendly conversation.")
        
        lines.append("")
        
        # Add context strategically
        if mode['use_context'] and history:
            # Only add relevant context
            context_weight = mode['context_weight']
            
            if context_weight > 0.6:  # High relevance - include more context
                relevant_history = history[-2:]
            else:  # Medium relevance - minimal context
                relevant_history = history[-1:]
            
            for user_msg, bot_msg in relevant_history:
                lines.append(f"Human: {user_msg.strip()}")
                lines.append(f"Assistant: {bot_msg.strip()}")
        
        # Add current exchange
        lines.append(f"Human: {user_input.strip()}")
        lines.append("Assistant:")
        
        return "\n".join(lines)
    
    def generate_response(self, user_input: str, history: List[Tuple[str, str]], 
                         emotion_label: str = "neutral", 
                         temperature: float = 0.7, max_new_tokens: int = 30) -> str:
        """
        Generate contextually intelligent responses
        """
        # Build intelligent prompt
        prompt = self.build_intelligent_prompt(user_input, history, emotion_label)
        
        # Get conversation mode for generation parameters
        mode = self.determine_conversation_mode(user_input, history)
        
        # Adjust generation parameters based on conversation mode
        if mode['is_question']:
            # More focused for questions
            temperature = min(temperature, 0.6)
            top_p = 0.8
        elif mode['is_emotional']:
            # More empathetic variance
            temperature = max(temperature, 0.8)
            top_p = 0.9
        else:
            # Standard conversation
            top_p = 0.85
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=350,  # Smaller context window
                padding=True,
                return_attention_mask=True
            )
            
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            seq_len = input_ids.shape[-1]
            max_length = min(seq_len + max_new_tokens, 400)
            
            # Generate with context-aware parameters
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=seq_len + 5,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    use_cache=True
                )
            
            # Decode new tokens
            new_tokens = output[0][seq_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean response
            cleaned_response = self._clean_response(response, user_input, mode)
            
            # Update conversation memory
            if cleaned_response:
                self.conversation_memory.append((user_input, cleaned_response))
                # Keep memory manageable
                if len(self.conversation_memory) > 10:
                    self.conversation_memory = self.conversation_memory[-8:]
            
            return cleaned_response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self._get_contextual_fallback(user_input, mode, emotion_label)
    
    def _clean_response(self, response: str, user_input: str, mode: Dict) -> str:
        """
        Clean response with context awareness
        """
        if not response:
            return ""
        
        # Remove artifacts
        artifacts = ["Human:", "Assistant:", "User:", "<|", "|>", "�"]
        for artifact in artifacts:
            if artifact in response:
                response = response.split(artifact)[0]
        
        # Clean up
        response = re.sub(r'\s+', ' ', response).strip()
        response = re.sub(r'^[.,!?;:]+', '', response).strip()
        
        # Context-aware cleaning
        if mode['is_question']:
            # For questions, ensure we have a complete answer
            if len(response) < 10 and not response.endswith(('.', '!', '?')):
                return ""
        elif mode['is_emotional']:
            # For emotional contexts, ensure empathetic response
            if len(response) < 5:
                return ""
        
        # Final validation
        if (len(response) < 3 or 
            'haha' in response.lower() and 'what' in user_input.lower() or
            response.lower().startswith(('um', 'uh'))):
            return ""
        
        return response.strip()
    
    def _get_contextual_fallback(self, user_input: str, mode: Dict, emotion_label: str) -> str:
        """
        Provide intelligent fallbacks based on conversation context
        """
        if mode['is_greeting']:
            return "Hello! How can I help you today?"
        elif mode['is_question']:
            return "That's a great question. Let me think about that."
        elif mode['is_emotional']:
            emotion_responses = {
                "sad": "I hear that you're going through a difficult time. I'm here to listen.",
                "happy": "I'm glad to hear you're feeling positive! That's wonderful.",
                "frustrated": "It sounds like you're feeling frustrated. That can be really tough.",
                "angry": "I can sense you're upset. It's okay to feel angry sometimes.",
                "confused": "It's completely normal to feel confused. Would you like to talk through it?"
            }
            return emotion_responses.get(emotion_label, "I understand how you're feeling. Tell me more.")
        elif mode['use_context']:
            return "I see what you mean. Can you tell me more about that?"
        else:
            return "That's interesting. What made you think of that?"
    
    def get_conversation_insights(self, history: List[Tuple[str, str]]) -> Dict:
        """
        Analyze conversation patterns for debugging
        """
        if not history:
            return {"total_exchanges": 0, "avg_relevance": 0.0}
        
        relevance_scores = []
        for i in range(1, len(history)):
            current_input = history[i][0]
            prev_history = history[:i]
            score = self.calculate_context_relevance(current_input, prev_history)
            relevance_scores.append(score)
        
        return {
            "total_exchanges": len(history),
            "avg_relevance": np.mean(relevance_scores) if relevance_scores else 0.0,
            "high_context_exchanges": sum(1 for score in relevance_scores if score > 0.5),
            "recent_relevance": relevance_scores[-3:] if len(relevance_scores) >= 3 else relevance_scores
        }