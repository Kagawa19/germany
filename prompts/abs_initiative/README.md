# ABS Initiative Prompts

This directory contains prompt templates for the ABS Initiative Metadata Explorer.

## Directory Structure

- `extraction/`: Prompts for web content extraction
- `analysis/`: Prompts for content analysis (sentiment, themes, geography, etc.)
- `quality/`: Prompts for quality assessment

## Usage

1. Prompts are loaded by the `PromptManager` from `prompt_manager.py`
2. Use `{placeholder}` syntax for variables that will be replaced at runtime
3. Keep prompts focused and specific to the ABS Initiative domain
4. Use JSON output formats when structured data is required

## Adding New Prompts

1. Create a new text file in the appropriate subdirectory
2. Use descriptive names (e.g., `organizations.txt`, `project_details.txt`)
3. Include clear instructions and expected output format
4. Document any placeholder variables that need to be replaced

## Testing Prompts

Use the `prompt_manager.py` utility to test prompts:

```python
from monitoring.prompt_manager import get_prompt_manager

# Get the prompt manager
pm = get_prompt_manager()

# Format a prompt with variables
formatted_prompt = pm.format_prompt(
    prompt_name="analysis/sentiment",
    variables={"content": "Sample text to analyze"},
    model="gpt-3.5-turbo"
)

print(formatted_prompt)
```
