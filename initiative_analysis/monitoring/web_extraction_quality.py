"""
Quality assessment integration for web extraction using JudgeAgent.
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents.judge import JudgeAgent
from monitoring.langfuse_client import get_langfuse_client

logger = logging.getLogger("quality_assessment")

class ExtractionQualityAssessor:
    """
    Class for assessing the quality of web extraction results using JudgeAgent.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the quality assessor with necessary components.
        
        Args:
            openai_api_key: Optional OpenAI API key for the judge
        """
        # Initialize Langfuse tracer
        self.langfuse = get_langfuse_client()
        
        # Initialize the JudgeAgent
        self.judge = self._initialize_judge(openai_api_key)
        
        # Set evaluation criteria
        self.evaluation_criteria = [
            "relevance",        # How relevant is the content to ABS Initiative
            "completeness",     # How complete is the extracted information
            "accuracy",         # How accurate is the analysis
            "usefulness"        # How useful is the information for the target audience
        ]
        
        logger.info("ExtractionQualityAssessor initialized")

    def _initialize_judge(self, openai_api_key: Optional[str] = None) -> Optional[JudgeAgent]:
        """
        Initialize the JudgeAgent for quality assessment.
        
        Args:
            openai_api_key: Optional OpenAI API key for the judge
            
        Returns:
            JudgeAgent instance or None if initialization fails
        """
        try:
            # Create the JudgeAgent
            judge = JudgeAgent(
                agent_id="abs_extraction_judge",
                tracer=self.langfuse,
                openai_tool=None  # We'll pass the API key in the config
            )
            
            # Configure the judge
            judge_config = {
                "evaluation_criteria": self.evaluation_criteria,
                "scoring_system": "scale_1_to_10",
                "detailed_feedback": True,
                "model": "gpt-4",  # Use GPT-4 for more accurate assessment
                "temperature": 0.2,
                "max_tokens": 1000,
                "openai_api_key": openai_api_key
            }
            
            # Initialize the judge
            judge.initialize(judge_config)
            
            logger.info("JudgeAgent initialized successfully")
            return judge
            
        except Exception as e:
            logger.error(f"Failed to initialize JudgeAgent: {str(e)}")
            return None

    async def assess_extraction_quality(self, 
                                      extraction_result: Dict[str, Any], 
                                      trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess the quality of an extraction result using the JudgeAgent.
        
        Args:
            extraction_result: The extraction result to assess
            trace_id: Optional Langfuse trace ID for monitoring
            
        Returns:
            Quality assessment results
        """
        # Check if JudgeAgent is available
        if not self.judge:
            logger.error("JudgeAgent not available for quality assessment")
            return {
                "error": "JudgeAgent not available",
                "score": 0.0,
                "status": "error"
            }
        
        # Create a span for quality assessment
        quality_span = None
        if self.langfuse and trace_id:
            quality_span = self.langfuse.create_span(
                trace_id=trace_id,
                name="assess_extraction_quality",
                metadata={
                    "result_count": len(extraction_result.get("results", [])),
                    "extraction_status": extraction_result.get("status", "unknown")
                }
            )
        
        try:
            # Create a query describing what we're evaluating
            query = f"""
            Evaluate the quality of this web extraction result for the ABS Initiative:
            - Language: {extraction_result.get('language', 'English')}
            - Status: {extraction_result.get('status', 'unknown')}
            - Result count: {len(extraction_result.get('results', []))}
            - Execution time: {extraction_result.get('execution_time', 'unknown')}
            
            The extraction should find relevant content about the Access and Benefit Sharing (ABS) Initiative
            and provide comprehensive metadata including sentiment, themes, geographic focus, etc.
            """
            
            # Format the input for JudgeAgent
            judge_input = {
                "query": query,
                "response": extraction_result,
                "criteria": self.evaluation_criteria
            }
            
            # Log assessment start
            logger.info(f"Starting quality assessment for extraction with {len(extraction_result.get('results', []))} results")
            start_time = time.time()
            
            # Execute the judgment
            judgment_result = await self.judge.async_execute(judge_input)
            
            # Calculate assessment duration
            assessment_duration = time.time() - start_time
            
            # Extract core judgment data
            judgment = judgment_result.get("judgment", {})
            overall_score = judgment.get("overall_score", 0)
            feedback = judgment.get("feedback", "No feedback available")
            criteria_scores = judgment.get("criteria_scores", {})
            
            # Log scores in Langfuse
            if self.langfuse and trace_id:
                # Log overall score
                self.langfuse.score(
                    trace_id=trace_id,
                    name="extraction_quality",
                    value=overall_score,
                    comment=f"Overall extraction quality score"
                )
                
                # Log individual criteria scores
                for criterion, score in criteria_scores.items():
                    self.langfuse.score(
                        trace_id=trace_id,
                        name=f"criterion_{criterion.lower()}",
                        value=score,
                        comment=f"Score for {criterion}"
                    )
            
            # Update quality span with results
            if quality_span:
                quality_span.update(
                    output={
                        "overall_score": overall_score,
                        "criteria_scores": criteria_scores,
                        "duration_seconds": assessment_duration
                    }
                )
                quality_span.end()
            
            # Log assessment completion
            logger.info(f"Quality assessment completed with overall score: {overall_score}/10")
            
            # Return formatted assessment results
            return {
                "overall_score": overall_score,
                "criteria_scores": criteria_scores,
                "feedback": feedback,
                "strengths": judgment.get("strengths", []),
                "improvements": judgment.get("improvements", []),
                "status": "completed",
                "duration_seconds": assessment_duration
            }
            
        except Exception as e:
            logger.error(f"Error assessing extraction quality: {str(e)}")
            
            # Update quality span with error
            if quality_span:
                quality_span.update(
                    output={
                        "error": str(e)
                    },
                    status="error"
                )
                quality_span.end()
            
            return {
                "error": str(e),
                "score": 0.0,
                "status": "error"
            }

    def assess_extraction_sync(self, 
                             extraction_result: Dict[str, Any], 
                             trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for assess_extraction_quality.
        
        Args:
            extraction_result: The extraction result to assess
            trace_id: Optional Langfuse trace ID for monitoring
            
        Returns:
            Quality assessment results
        """
        return asyncio.run(self.assess_extraction_quality(extraction_result, trace_id))


# Usage example in your Streamlit app:
"""
from quality_assessment import ExtractionQualityAssessor

# In your web extraction handler:
def handle_extraction():
    # Initialize the extractor
    extractor = WebExtractor(language="English")
    
    # Create Langfuse trace
    langfuse = get_langfuse_client()
    trace = langfuse.create_trace(name="web_extraction", metadata={"source": "streamlit"})
    
    # Run extraction
    extraction_result = extractor.run(max_queries=5, max_results_per_query=10)
    
    # If successful and results found, assess quality
    if extraction_result.get("status") == "success" and extraction_result.get("result_count", 0) > 0:
        # Initialize quality assessor
        assessor = ExtractionQualityAssessor()
        
        # Run quality assessment
        quality_result = assessor.assess_extraction_sync(extraction_result, trace.id)
        
        # Add quality info to result
        extraction_result["quality_assessment"] = quality_result
        
        # Display quality score to user
        st.success(f"Extraction quality score: {quality_result.get('overall_score', 0)}/10")
        
        # Show feedback
        st.write("Quality feedback:")
        st.write(quality_result.get("feedback", "No feedback available"))
    
    return extraction_result
"""