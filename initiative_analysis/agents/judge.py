"""
Judge implementation for content quality assessment.
Evaluates extraction and analysis results using LLM-based judgment.
"""

import logging
import time
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from monitoring.langfuse_client import get_langfuse_client

logger = logging.getLogger(__name__)

class Judge:
    """
    Class that evaluates results using an LLM to make judgments.
    Designed for assessing content quality and analysis results.
    """
    
    def __init__(
        self,
        judge_id: str,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the Judge with the necessary components.
        
        Args:
            judge_id: Unique ID for the judge
            openai_api_key: OpenAI API key for LLM judgments
        """
        # Store basic configuration
        self.judge_id = judge_id
        self.openai_api_key = openai_api_key
        
        # Initialize Langfuse tracer
        self.langfuse = get_langfuse_client()
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai()
        
        # Set default configurations
        self.evaluation_criteria = ["relevance", "accuracy", "completeness", "usefulness"]
        self.model = "gpt-4" if self.openai_client else None
        self.temperature = 0.2
        self.max_tokens = 1000
        
        # Mark as not fully initialized until configure() is called
        self.initialized = False
        
        logger.info(f"Judge '{judge_id}' instantiated")

    def _initialize_openai(self):
        """
        Initialize the OpenAI client.
        
        Returns:
            OpenAI client instance or None if initialization fails
        """
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Judge capabilities will be limited.")
            return None
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            return client
        except ImportError:
            logger.error("OpenAI package not installed. Please install with 'pip install openai'")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the judge with custom settings.
        
        Args:
            config: Configuration dictionary with settings
            
        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            # Update API key if provided
            if "openai_api_key" in config and config["openai_api_key"]:
                self.openai_api_key = config["openai_api_key"]
                # Re-initialize OpenAI client with new key
                self.openai_client = self._initialize_openai()
            
            # Update evaluation settings
            if "evaluation_criteria" in config:
                self.evaluation_criteria = config["evaluation_criteria"]
                
            if "model" in config:
                self.model = config["model"]
                
            if "temperature" in config:
                self.temperature = config["temperature"]
                
            if "max_tokens" in config:
                self.max_tokens = config["max_tokens"]
            
            # Load custom prompts if provided
            if "prompt_template" in config:
                self.prompt_template = config["prompt_template"]
            else:
                # Use default prompt template
                self.prompt_template = self._get_default_prompt_template()
            
            # Mark as initialized
            self.initialized = True
            
            logger.info(f"Judge '{self.judge_id}' configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure judge: {str(e)}")
            return False

    def _get_default_prompt_template(self) -> str:
        """
        Get the default prompt template for evaluation.
        
        Returns:
            Default prompt template string
        """
        return """
        # Task: Evaluate the quality of extracted and analyzed content

        ## Query/Source:
        {query}

        ## Extracted Content and Analysis to Evaluate:
        {response}

        ## Evaluation Criteria:
        {criteria_str}

        ## Instructions:
        1. Carefully analyze the extracted content and analysis against the original query/source.
        2. Evaluate how well the extraction and analysis address each criterion.
        3. Provide a score from 1-10 for each criterion (10 being perfect).
        4. Calculate an overall score from 1-10.
        5. Provide specific, constructive feedback highlighting strengths and areas for improvement.
        6. Respond ONLY with a valid JSON in the following structure:
        
        {
        "criteria_scores": {
            "criterion1": number,
            "criterion2": number
        },
        "overall_score": number,
        "feedback": "string",
        "strengths": ["string"],
        "improvements": ["string"]
        }
        """

    async def evaluate_async(self, 
                           query: str, 
                           response: Any,
                           criteria: Optional[List[str]] = None,
                           trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate content asynchronously using LLM.
        
        Args:
            query: Original query or source description
            response: Content to evaluate (can be string or JSON object)
            criteria: Custom criteria to use for this evaluation
            trace_id: Optional Langfuse trace ID for monitoring
            
        Returns:
            Evaluation results dictionary
        """
        # Validate initialization
        if not self.initialized:
            logger.error("Judge not initialized. Call configure() first.")
            return {
                "error": "Judge not initialized",
                "overall_score": 0
            }
        
        # Create evaluation span
        eval_span = None
        if self.langfuse and trace_id:
            eval_span = self.langfuse.create_span(
                trace_id=trace_id,
                name=f"judge_evaluation_{self.judge_id}",
                metadata={
                    "judge_id": self.judge_id,
                    "model": self.model
                }
            )
        
        # Use provided criteria or default
        evaluation_criteria = criteria or self.evaluation_criteria
        
        # Format criteria for prompt
        criteria_str = "\n".join([f"- {c.title()}" for c in evaluation_criteria])
        
        try:
            # Check if OpenAI client is available
            if not self.openai_client:
                error_msg = "OpenAI client not available for evaluation"
                logger.error(error_msg)
                
                # Update evaluation span with error
                if eval_span:
                    eval_span.update(
                        output={"error": error_msg},
                        status="error"
                    )
                    eval_span.end()
                    
                return {
                    "error": error_msg,
                    "overall_score": 0
                }
            
            # Prepare the prompt
            prompt = self.prompt_template.format(
                query=query,
                response=json.dumps(response) if isinstance(response, dict) else response,
                criteria_str=criteria_str
            )
            
            # Log the evaluation request
            logger.info(f"Evaluating content with judge '{self.judge_id}'")
            
            # Track the start time
            start_time = time.time()
            
            # Make the OpenAI API call
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond with ONLY a JSON object that precisely matches the specified JSON structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract the response text
            evaluation_text = response.choices[0].message.content.strip()
            evaluation_duration = time.time() - start_time
            
            # Log the LLM generation for this evaluation
            if self.langfuse and trace_id:
                self.langfuse.log_generation(
                    trace_id=trace_id,
                    name="judge_evaluation",
                    model=self.model,
                    prompt=prompt,
                    completion=evaluation_text,
                    metadata={
                        "duration": evaluation_duration,
                        "criteria": evaluation_criteria
                    },
                    parent_span_id=eval_span.id if eval_span else None
                )
            
            # Parse the response
            try:
                evaluation_data = json.loads(evaluation_text)
                
                # Validate and add default values if needed
                if "overall_score" not in evaluation_data:
                    evaluation_data["overall_score"] = 5
                
                if "criteria_scores" not in evaluation_data:
                    evaluation_data["criteria_scores"] = {c: 5 for c in evaluation_criteria}
                
                if "feedback" not in evaluation_data:
                    evaluation_data["feedback"] = "No feedback provided"
                
                if "strengths" not in evaluation_data:
                    evaluation_data["strengths"] = []
                
                if "improvements" not in evaluation_data:
                    evaluation_data["improvements"] = []
                
                # Ensure overall_score is within 0-10 range
                evaluation_data["overall_score"] = max(0, min(10, evaluation_data["overall_score"]))
                
                # Log scores in Langfuse
                if self.langfuse and trace_id:
                    # Log overall score
                    self.langfuse.score(
                        trace_id=trace_id,
                        name="judge_overall_score",
                        value=evaluation_data["overall_score"],
                        comment=f"Overall quality score"
                    )
                    
                    # Log individual criteria scores
                    for criterion, score in evaluation_data["criteria_scores"].items():
                        self.langfuse.score(
                            trace_id=trace_id,
                            name=f"judge_criterion_{criterion.lower()}",
                            value=score,
                            comment=f"Score for {criterion}"
                        )
                
                # Update evaluation span with results
                if eval_span:
                    eval_span.update(
                        output={
                            "status": "success",
                            "overall_score": evaluation_data["overall_score"],
                            "criteria_scores": evaluation_data["criteria_scores"],
                            "duration_seconds": evaluation_duration
                        }
                    )
                    eval_span.end()
                
                # Log completion
                logger.info(f"Evaluation completed with overall score: {evaluation_data['overall_score']}/10")
                
                return evaluation_data
                
            except json.JSONDecodeError:
                error_msg = "Failed to parse evaluation response as JSON"
                logger.error(f"{error_msg}: {evaluation_text[:100]}...")
                
                # Update evaluation span with error
                if eval_span:
                    eval_span.update(
                        output={
                            "error": error_msg,
                            "raw_response": evaluation_text[:500]
                        },
                        status="error"
                    )
                    eval_span.end()
                
                return {
                    "error": error_msg,
                    "raw_response": evaluation_text,
                    "overall_score": 0
                }
                
        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Update evaluation span with error
            if eval_span:
                eval_span.update(
                    output={"error": error_msg},
                    status="error"
                )
                eval_span.end()
            
            return {
                "error": error_msg,
                "overall_score": 0
            }

    def evaluate(self, 
                query: str, 
                response: Any,
                criteria: Optional[List[str]] = None,
                trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_async.
        
        Args:
            query: Original query or source description
            response: Content to evaluate (can be string or JSON object)
            criteria: Custom criteria to use for this evaluation
            trace_id: Optional Langfuse trace ID for monitoring
            
        Returns:
            Evaluation results dictionary
        """
        return asyncio.run(self.evaluate_async(query, response, criteria, trace_id))