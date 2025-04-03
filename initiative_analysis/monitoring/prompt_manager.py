"""
Prompt manager for tracking and loading prompts.
Provides centralized prompt management with versioning and logging.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from monitoring.langfuse_client import get_langfuse_client

# Configure logging
logger = logging.getLogger("prompt_manager")

class PromptManager:
    """
    Manager for tracking, loading and versioning prompts.
    """
    
    def __init__(self, prompts_dir: str = "prompts/abs_initiative"):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        # Set prompts directory
        self.prompts_dir = prompts_dir
        
        # Initialize prompts storage
        self.prompts = {}
        self.prompt_usage = {}
        
        # Initialize Langfuse client for tracking
        self.langfuse = get_langfuse_client()
        
        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Load all prompts
        self._load_all_prompts()
        
        logger.info(f"PromptManager initialized with {len(self.prompts)} prompts from {self.prompts_dir}")

    def _load_all_prompts(self):
        """
        Load all prompts from the prompts directory.
        """
        try:
            # Get all txt files in the prompts directory
            prompt_files = Path(self.prompts_dir).glob("*.txt")
            
            for file_path in prompt_files:
                prompt_name = file_path.stem
                
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                
                # Store the prompt
                self.prompts[prompt_name] = prompt_text
                self.prompt_usage[prompt_name] = 0
                
                logger.debug(f"Loaded prompt: {prompt_name}")
                
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Get a prompt by name.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Prompt text or None if not found
        """
        # If prompt is already loaded, return it
        if prompt_name in self.prompts:
            # Increment usage counter
            self.prompt_usage[prompt_name] += 1
            return self.prompts[prompt_name]
        
        # Try to load from file
        try:
            prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
            
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                
                # Store the prompt
                self.prompts[prompt_name] = prompt_text
                self.prompt_usage[prompt_name] = 1
                
                logger.info(f"Loaded prompt on demand: {prompt_name}")
                
                return prompt_text
            else:
                logger.warning(f"Prompt not found: {prompt_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {str(e)}")
            return None

    def track_prompt_usage(self, 
                          prompt_name: str, 
                          model: str, 
                          input_text: Optional[str] = None,
                          output_text: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          trace_id: Optional[str] = None) -> None:
        """
        Track prompt usage in Langfuse.
        
        Args:
            prompt_name: Name of the prompt
            model: Model used for generation
            input_text: Input text for the generation
            output_text: Output text from the generation
            metadata: Additional metadata
            trace_id: Optional Langfuse trace ID
        """
        if not self.langfuse.enabled:
            return
        
        if not prompt_name or prompt_name not in self.prompts:
            logger.warning(f"Cannot track unknown prompt: {prompt_name}")
            return
        
        try:
            # Get the prompt template
            prompt_template = self.prompts[prompt_name]
            
            # Log the generation
            self.langfuse.log_generation(
                trace_id=trace_id,
                name=f"prompt_{prompt_name}",
                model=model,
                prompt=input_text or prompt_template,
                completion=output_text or "",
                metadata={
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template[:500] + "..." if len(prompt_template) > 500 else prompt_template,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
            
            logger.info(f"Tracked usage of prompt: {prompt_name}")
            
        except Exception as e:
            logger.error(f"Error tracking prompt usage: {str(e)}")

    def format_prompt(self, 
                     prompt_name: str, 
                     variables: Dict[str, Any],
                     model: str = "gpt-3.5-turbo",
                     track: bool = True,
                     trace_id: Optional[str] = None) -> Optional[str]:
        """
        Get and format a prompt with variables.
        
        Args:
            prompt_name: Name of the prompt
            variables: Dictionary of variables to format the prompt
            model: Model that will use this prompt
            track: Whether to track this prompt usage
            trace_id: Optional Langfuse trace ID
            
        Returns:
            Formatted prompt or None if not found
        """
        # Get the prompt template
        prompt_template = self.get_prompt(prompt_name)
        
        if not prompt_template:
            return None
        
        try:
            # Format the prompt
            formatted_prompt = prompt_template.format(**variables)
            
            # Track usage if requested
            if track:
                self.track_prompt_usage(
                    prompt_name=prompt_name,
                    model=model,
                    input_text=formatted_prompt,
                    metadata={"variables": {k: str(v)[:100] for k, v in variables.items()}},
                    trace_id=trace_id
                )
            
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"Missing variable in prompt {prompt_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error formatting prompt {prompt_name}: {str(e)}")
            return None

    def save_prompt(self, 
                   prompt_name: str, 
                   prompt_text: str, 
                   overwrite: bool = False) -> bool:
        """
        Save a new prompt or update an existing one.
        
        Args:
            prompt_name: Name of the prompt
            prompt_text: Prompt text content
            overwrite: Whether to overwrite existing prompt
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
            
            # Check if prompt already exists
            if os.path.exists(prompt_path) and not overwrite:
                logger.warning(f"Prompt {prompt_name} already exists. Use overwrite=True to update.")
                return False
            
            # Save the prompt
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            
            # Update in-memory storage
            self.prompts[prompt_name] = prompt_text
            if prompt_name not in self.prompt_usage:
                self.prompt_usage[prompt_name] = 0
            
            logger.info(f"Saved prompt: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prompt {prompt_name}: {str(e)}")
            return False

    def get_prompt_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics on prompt usage.
        
        Returns:
            Dictionary with prompt usage counts
        """
        return dict(self.prompt_usage)
    
    def get_all_prompt_names(self) -> list:
        """
        Get all available prompt names.
        
        Returns:
            List of prompt names
        """
        return list(self.prompts.keys())


# Helper function to get a singleton instance
_prompt_manager_instance = None

def get_prompt_manager(prompts_dir: Optional[str] = None) -> PromptManager:
    """
    Get or create the singleton PromptManager instance.
    
    Args:
        prompts_dir: Optional custom prompts directory
        
    Returns:
        PromptManager instance
    """
    global _prompt_manager_instance
    
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager(prompts_dir or "prompts/abs_initiative")
    elif prompts_dir is not None and prompts_dir != _prompt_manager_instance.prompts_dir:
        # If a different prompts_dir is specified, create a new instance
        _prompt_manager_instance = PromptManager(prompts_dir)
        
    return _prompt_manager_instance