"""
Langfuse client initialization and configuration.
Provides a singleton client for tracing and monitoring throughout the app.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from functools import lru_cache

# Import langfuse
from langfuse import Langfuse
from langfuse.client import Trace, Span, Generation

# Configure logging
logger = logging.getLogger("langfuse_monitoring")

class LangfuseClient:
    """Singleton class for Langfuse client management"""
    
    _instance = None

    def __new__(cls):
        """Ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(LangfuseClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Langfuse client with credentials from environment"""
        if self._initialized:
            return
            
        # Load environment variables
        load_dotenv()
        
        # Get Langfuse credentials from environment
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # Check if credentials are available
        if not self.public_key or not self.secret_key:
            logger.warning("Langfuse credentials not found in environment. Monitoring will be disabled.")
            self.client = None
            self.enabled = False
        else:
            # Initialize Langfuse client
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                self.enabled = True
                logger.info(f"Langfuse client initialized successfully. Host: {self.host}")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse client: {str(e)}")
                self.client = None
                self.enabled = False
        
        self._initialized = True

    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None, 
                    tags: Optional[list] = None, user_id: Optional[str] = None) -> Optional[Trace]:
        """
        Create a new trace for a complete process flow.
        
        Args:
            name: Name of the trace
            metadata: Optional metadata to include
            tags: Optional tags for categorization
            user_id: Optional user identifier
            
        Returns:
            Trace object or None if client is disabled
        """
        if not self.enabled or not self.client:
            return None
            
        try:
            return self.client.trace(
                name=name,
                metadata=metadata or {},
                tags=tags or [],
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Error creating trace '{name}': {str(e)}")
            return None

    def create_span(self, trace_id: Optional[str] = None, name: str = "unnamed_span",
                   metadata: Optional[Dict[str, Any]] = None, 
                   parent_span_id: Optional[str] = None) -> Optional[Span]:
        """
        Create a span within a trace for tracking sub-operations.
        
        Args:
            trace_id: Optional ID of the parent trace
            name: Name of the span
            metadata: Optional metadata to include
            parent_span_id: Optional ID of parent span for nesting
            
        Returns:
            Span object or None if client is disabled
        """
        if not self.enabled or not self.client:
            return None
            
        try:
            return self.client.span(
                trace_id=trace_id,
                name=name,
                metadata=metadata or {},
                parent_span_id=parent_span_id
            )
        except Exception as e:
            logger.error(f"Error creating span '{name}': {str(e)}")
            return None

    def log_generation(self, trace_id: Optional[str] = None, name: str = "unnamed_generation",
                     model: str = "gpt-3.5-turbo", prompt: Optional[str] = None,
                     completion: Optional[str] = None, 
                     metadata: Optional[Dict[str, Any]] = None,
                     parent_span_id: Optional[str] = None) -> Optional[Generation]:
        """
        Log an LLM generation event within a trace.
        
        Args:
            trace_id: Optional ID of the parent trace
            name: Name of the generation event
            model: Model used for generation
            prompt: The prompt sent to the model
            completion: The model's response
            metadata: Optional metadata to include
            parent_span_id: Optional ID of parent span
            
        Returns:
            Generation object or None if client is disabled
        """
        if not self.enabled or not self.client:
            return None
            
        try:
            return self.client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                prompt=prompt,
                completion=completion,
                metadata=metadata or {},
                parent_span_id=parent_span_id
            )
        except Exception as e:
            logger.error(f"Error logging generation '{name}': {str(e)}")
            return None

    def score(self, trace_id: Optional[str] = None, name: str = "quality_score", 
             value: float = 0.0, comment: Optional[str] = None) -> None:
        """
        Log a score for quality assessment.
        
        Args:
            trace_id: Optional ID of the parent trace
            name: Name of the score metric
            value: Numeric score value
            comment: Optional comment explaining the score
            
        Returns:
            None
        """
        if not self.enabled or not self.client:
            return
            
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
            logger.debug(f"Logged score '{name}': {value}")
        except Exception as e:
            logger.error(f"Error logging score '{name}': {str(e)}")

    def flush(self) -> None:
        """
        Manually flush all queued events to Langfuse.
        Should be called when application exits.
        
        Returns:
            None
        """
        if not self.enabled or not self.client:
            return
            
        try:
            self.client.flush()
            logger.debug("Flushed Langfuse client")
        except Exception as e:
            logger.error(f"Error flushing Langfuse client: {str(e)}")


@lru_cache(maxsize=1)
def get_langfuse_client() -> LangfuseClient:
    """
    Get or create a singleton Langfuse client.
    Uses lru_cache to ensure a single instance is returned.
    
    Returns:
        LangfuseClient instance
    """
    return LangfuseClient()