#!/bin/bash

# Function to create or overwrite a prompt file
create_prompt() {
    local filename="$1"
    local content="$2"
    
    echo "$content" > "prompts/$filename"
    echo "Populated $filename"
}

# Conciseness Prompt
create_prompt "conciseness.txt" "Evaluate the conciseness of the following text:

{{content}}

Conciseness Assessment Criteria:
- Efficiency of information delivery
- Absence of unnecessary repetition
- Clarity of expression
- Direct communication of key points

Respond in JSON format:
{{
\"conciseness_score\": 0.0-1.0,
\"conciseness_confidence\": 0.0-1.0,
\"conciseness_issues\": [
    \"Specific area of verbosity\",
    \"Redundant explanation\"
]
}}"

# Correctness Prompt
create_prompt "correctness.txt" "Evaluate the factual correctness of the following text:

{{content}}

Factual Correctness Assessment:
- Accuracy of statements
- Alignment with established information
- Precision of technical claims
- Consistency with current knowledge

Respond in JSON format:
{{
\"correctness_score\": 0.0-1.0,
\"correctness_confidence\": 0.0-1.0,
\"correctness_issues\": [
    \"Specific factual inaccuracy\",
    \"Potential misrepresentation\"
]
}}"

# Hallucination Prompt
create_prompt "hallucination.txt" "Analyze the following text for potential fabricated or unsupported information:

{{content}}

Hallucination Detection Criteria:
- Identification of potentially non-existent claims
- Detection of fabricated events or statistics
- Verification of unsupported assertions
- Recognition of implausible statements

Respond in JSON format:
{{
\"hallucination_score\": 0.0-1.0,
\"hallucination_confidence\": 0.0-1.0,
\"hallucination_issues\": [
    \"Specific hallucination or fabricated claim\",
    \"Unverifiable assertion\"
]
}}"

# Helpfulness Prompt
create_prompt "helpfulness.txt" "Evaluate the helpfulness of the following text:

{{content}}

Helpfulness Assessment Criteria:
- Practical value of information
- Clarity of explanations
- Actionable insights
- Potential usefulness to stakeholders

Respond in JSON format:
{{
\"helpfulness_score\": 0.0-1.0,
\"helpfulness_confidence\": 0.0-1.0,
\"helpfulness_notes\": [
    \"Specific helpful aspect\",
    \"Potential improvement area\"
]
}}"

# Relevance Prompt
create_prompt "relevance.txt" "Evaluate the overall relevance of the following text:

{{content}}

Relevance Assessment Criteria:
- Significance to current discussions
- Potential impact or importance
- Depth of insight
- Breadth of applicability

Respond in JSON format:
{{
\"relevance_score\": 0.0-1.0,
\"relevance_confidence\": 0.0-1.0,
\"relevance_key_points\": [
    \"Specific relevance indicator\",
    \"Another key relevance factor\"
]
}}"

# Toxicity Prompt
create_prompt "toxicity.txt" "Analyze the following text for potentially harmful language:

{{content}}

Toxicity Assessment Criteria:
- Presence of discriminatory language
- Harmful stereotypes or prejudices
- Inflammatory or aggressive tone
- Potential for causing offense

Respond in JSON format:
{{
\"toxicity_score\": 0.0-1.0,
\"toxicity_confidence\": 0.0-1.0,
\"toxicity_concerns\": [
    \"Specific toxic language or perspective\",
    \"Potential harmful implication\"
]
}}"

# Summary Validation Prompt
create_prompt "summary_validation.txt" "Validate the accuracy of the following summary against the original content:

Summary: {{summary}}
Original Content: {{content}}

Validation Criteria:
- Factual accuracy of summary claims
- Preservation of original content's key messages
- Absence of hallucinations or fabricated information
- Maintenance of original context and nuance

Respond in JSON format:
{{
\"valid\": true/false,
\"issues\": [
    \"Specific validation concern\",
    \"Potential misrepresentation\"
],
\"explanation\": \"Brief rationale for validation result\"
}}"

echo "Specified prompts populated with double-brace formatting!"