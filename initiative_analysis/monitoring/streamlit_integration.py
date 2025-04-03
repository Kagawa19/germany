"""
Streamlit monitoring integration for Langfuse.
Add this to the beginning of your main.py file.
"""

import streamlit as st
import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

from monitoring.langfuse_client import get_langfuse_client

# Configure logging
logger = logging.getLogger("streamlit_monitoring")

# Initialize Langfuse client
langfuse = get_langfuse_client()

# Get or create a session ID for the current user
def get_session_id():
    """Get a persistent session ID for the current user."""
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# Store the Langfuse trace for the current session
def get_session_trace():
    """Get or create a Langfuse trace for the current session."""
    if "langfuse_trace" not in st.session_state:
        # Create a new trace for this session
        trace = langfuse.create_trace(
            name="streamlit_session",
            user_id=get_session_id(),
            metadata={"source": "streamlit_app"}
        )
        st.session_state.langfuse_trace = trace
    return st.session_state.langfuse_trace

# Track page views
def track_page_view(page_name: str):
    """
    Track a page view in Langfuse.
    
    Args:
        page_name: Name of the page being viewed
    """
    if not langfuse.enabled:
        return
        
    try:
        # Get the session trace
        trace = get_session_trace()
        if not trace:
            return
            
        # Create a span for the page view
        span = langfuse.create_span(
            trace_id=trace.id,
            name=f"page_view_{page_name}",
            metadata={"page_name": page_name}
        )
        
        # Store the span in session state for later updates
        st.session_state[f"page_span_{page_name}"] = span
        
        logger.info(f"Tracked page view: {page_name}")
        
    except Exception as e:
        logger.error(f"Error tracking page view: {str(e)}")

# End page view tracking
def end_page_view(page_name: str):
    """
    End tracking for a page view.
    
    Args:
        page_name: Name of the page being viewed
    """
    if not langfuse.enabled:
        return
        
    try:
        # Get the span from session state
        span_key = f"page_span_{page_name}"
        if span_key in st.session_state:
            span = st.session_state[span_key]
            span.end()
            # Remove the span from session state
            del st.session_state[span_key]
            
            logger.info(f"Ended page view tracking: {page_name}")
            
    except Exception as e:
        logger.error(f"Error ending page view tracking: {str(e)}")

# Track user interactions
def track_interaction(interaction_type: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Track a user interaction in Langfuse.
    
    Args:
        interaction_type: Type of interaction (e.g., button_click, filter_change)
        metadata: Additional metadata for the interaction
    """
    if not langfuse.enabled:
        return
        
    try:
        # Get the session trace
        trace = get_session_trace()
        if not trace:
            return
            
        # Create a span for the interaction
        span = langfuse.create_span(
            trace_id=trace.id,
            name=f"interaction_{interaction_type}",
            metadata=metadata or {}
        )
        
        # End the span immediately for one-time interactions
        span.end()
        
        logger.info(f"Tracked user interaction: {interaction_type}")
        
    except Exception as e:
        logger.error(f"Error tracking interaction: {str(e)}")

# Decorator for timing Streamlit functions
def track_function(name: Optional[str] = None, track_args: bool = False):
    """
    Decorator to track execution of Streamlit functions.
    
    Args:
        name: Optional custom name for the function
        track_args: Whether to include function arguments in tracking
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langfuse.enabled:
                return func(*args, **kwargs)
                
            # Get function name
            func_name = name or func.__name__
            
            try:
                # Get the session trace
                trace = get_session_trace()
                if not trace:
                    return func(*args, **kwargs)
                    
                # Create metadata for the span
                metadata = {"function": func_name}
                
                # Add args if requested
                if track_args:
                    # Only include simple args to avoid serialization issues
                    safe_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            safe_kwargs[k] = v
                    metadata["args"] = safe_kwargs
                
                # Create a span for the function execution
                span = langfuse.create_span(
                    trace_id=trace.id,
                    name=f"function_{func_name}",
                    metadata=metadata
                )
                
                # Execute the function and measure time
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Update span with success
                    span.update(
                        output={
                            "status": "success",
                            "duration_seconds": duration
                        }
                    )
                    span.end()
                    
                    return result
                except Exception as e:
                    # Update span with error
                    span.update(
                        output={
                            "status": "error",
                            "error": str(e)
                        },
                        status="error"
                    )
                    span.end()
                    
                    # Re-raise the exception
                    raise
                    
            except Exception as e:
                logger.error(f"Error tracking function {func_name}: {str(e)}")
                # Fall back to just calling the function
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

# Tracking for Streamlit dashboard loading
def track_dashboard_load(dashboard_name: str):
    """
    Track dashboard loading performance.
    
    Args:
        dashboard_name: Name of the dashboard
    """
    if not langfuse.enabled:
        return
        
    try:
        # Get the session trace
        trace = get_session_trace()
        if not trace:
            return
            
        # Create a span for the dashboard load
        span = langfuse.create_span(
            trace_id=trace.id,
            name=f"dashboard_load_{dashboard_name}",
            metadata={"dashboard": dashboard_name}
        )
        
        # Store the span in session state
        st.session_state[f"dashboard_span_{dashboard_name}"] = span
        
        # Store the start time
        st.session_state[f"dashboard_start_{dashboard_name}"] = time.time()
        
        logger.info(f"Started tracking dashboard load: {dashboard_name}")
        
    except Exception as e:
        logger.error(f"Error tracking dashboard load: {str(e)}")

# End dashboard load tracking
def end_dashboard_load(dashboard_name: str, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
    """
    End tracking for dashboard loading.
    
    Args:
        dashboard_name: Name of the dashboard
        success: Whether loading was successful
        metadata: Additional metadata for the dashboard load
    """
    if not langfuse.enabled:
        return
        
    try:
        # Get the span and start time from session state
        span_key = f"dashboard_span_{dashboard_name}"
        start_key = f"dashboard_start_{dashboard_name}"
        
        if span_key in st.session_state and start_key in st.session_state:
            span = st.session_state[span_key]
            start_time = st.session_state[start_key]
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update and end the span
            span.update(
                output={
                    "status": "success" if success else "error",
                    "duration_seconds": duration,
                    **(metadata or {})
                },
                status="success" if success else "error"
            )
            span.end()
            
            # Remove from session state
            del st.session_state[span_key]
            del st.session_state[start_key]
            
            logger.info(f"Ended tracking dashboard load: {dashboard_name} in {duration:.2f}s")
            
    except Exception as e:
        logger.error(f"Error ending dashboard load tracking: {str(e)}")