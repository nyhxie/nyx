from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import logging

@dataclass
class ConversationSegment:
    messages: List[dict]
    topic: str
    start_time: datetime
    end_time: datetime
    summary: Optional[str] = None

class ConversationAnalyzer:
    def __init__(self):
        self.topic_model = LatentDirichletAllocation(n_components=5, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
    def detect_topic_shift(self, messages: List[dict], threshold: float = 0.3) -> List[ConversationSegment]:
        """Detect shifts in conversation topics and segment messages accordingly"""
        if not messages:
            return []
            
        # Validate and normalize messages
        normalized_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            content = msg.get("content")
            if not content or not isinstance(content, str):
                continue
                
            # Use provided timestamp or current time as fallback
            timestamp = msg.get("timestamp")
            if not timestamp:
                timestamp = datetime.now()
                
            normalized_messages.append({
                "content": content,
                "role": msg.get("role", "user"),
                "timestamp": timestamp
            })
            
        if not normalized_messages:
            return []
            
        # Convert messages to text documents
        docs = [msg["content"] for msg in normalized_messages]
        
        # Create TF-IDF matrix
        X = self.vectorizer.fit_transform(docs)
        
        # Get topic distributions for each message
        topic_distributions = self.topic_model.fit_transform(X)
        
        # Detect topic shifts
        segments = []
        current_segment = []
        current_topic = np.argmax(topic_distributions[0])
        segment_start = normalized_messages[0]["timestamp"]
        
        for i in range(len(normalized_messages)):
            msg = normalized_messages[i]
            msg_topic = np.argmax(topic_distributions[i])
            
            # Check if topic has shifted significantly
            if msg_topic != current_topic and np.max(topic_distributions[i]) > threshold:
                if current_segment:
                    segments.append(ConversationSegment(
                        messages=current_segment,
                        topic=self._get_topic_keywords(current_topic),
                        start_time=segment_start,
                        end_time=normalized_messages[i-1]["timestamp"]
                    ))
                current_segment = []
                segment_start = msg["timestamp"]
                current_topic = msg_topic
            
            current_segment.append(msg)
        
        # Add final segment
        if current_segment:
            segments.append(ConversationSegment(
                messages=current_segment,
                topic=self._get_topic_keywords(current_topic),
                start_time=segment_start,
                end_time=current_segment[-1]["timestamp"]
            ))
        
        return segments
    
    def summarize_segment(self, segment: ConversationSegment) -> str:
        """Generate a summary for a conversation segment"""
        try:
            # Concatenate messages with speaker identification
            text = " ".join([
                f"{msg['role']}: {msg['content']}" for msg in segment.messages
            ])
            
            # If text is too short, return it as is
            if len(text) < 100:
                segment.summary = text
                return text
                
            # Calculate dynamic max_length based on input length
            input_length = len(text.split())
            max_length = min(130, max(30, input_length // 2))
            min_length = min(30, max_length - 20)
            
            # Generate summary with adjusted parameters
            try:
                summary = self.summarizer(
                    text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False,
                    truncation=True
                )[0]["summary_text"]
            except Exception as e:
                logging.warning(f"Summarization failed with error: {e}")
                # Fallback: use first sentence or truncate
                sentences = text.split('.')
                summary = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
                
            segment.summary = summary
            return summary
            
        except Exception as e:
            logging.error(f"Error in summarize_segment: {e}")
            # Return a safe fallback
            fallback = "Error generating summary"
            segment.summary = fallback
            return fallback
    
    def _get_topic_keywords(self, topic_idx: int, num_words: int = 5) -> str:
        """Get the top keywords representing a topic"""
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = self.topic_model.components_[topic_idx]
        top_words = [feature_names[i] for i in topic_words.argsort()[:-num_words-1:-1]]
        return ", ".join(top_words)