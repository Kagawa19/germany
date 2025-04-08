#!/bin/bash

# Function to create prompt file
create_prompt() {
    local filename="$1"
    local content="$2"
    
    echo "$content" > "prompts/$filename"
    echo "Populated $filename"
}

# Conciseness Prompt
create_prompt "conciseness.txt" "Evaluate the conciseness of the following text excerpt.

Content: {content}

Conciseness Assessment Criteria:
- Efficiency of information delivery
- Absence of unnecessary repetition
- Clarity of expression
- Direct communication of key points

Specific Considerations:
- Are ideas expressed succinctly?
- Is there unnecessary elaboration?
- Do sentences serve a clear purpose?
- Is the language direct and to the point?

Respond in JSON format:
{
  \"conciseness_score\": 0.0-1.0,
  \"conciseness_confidence\": 0.0-1.0,
  \"issues\": [
    \"Specific area of verbosity\",
    \"Redundant explanation\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: Extremely verbose and inefficient
- 0.3-0.5: Significant room for improvement
- 0.6-0.8: Generally concise with some areas to tighten
- 0.9-1.0: Extremely efficient and direct communication"

# Correctness Prompt
create_prompt "correctness.txt" "Evaluate the factual correctness of the following text excerpt with a focus on Access and Benefit Sharing (ABS) and related topics.

Content: {content}

Factual Correctness Assessment Criteria:
- Accuracy of statements about ABS concepts
- Alignment with established scientific and policy information
- Precision of technical and scientific claims
- Consistency with current knowledge in biodiversity and genetic resources

Specific Evaluation Points:
- Verify claims against known scientific and policy facts
- Check for potential misrepresentations of ABS principles
- Assess the reliability of statistical or numerical claims
- Examine the accuracy of references to international protocols

Respond in JSON format:
{
  \"correctness_score\": 0.0-1.0,
  \"correctness_confidence\": 0.0-1.0,
  \"issues\": [
    \"Specific factual inaccuracy\",
    \"Potential misrepresentation\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: Contains multiple significant factual errors
- 0.3-0.5: Several inaccuracies or misrepresentations
- 0.6-0.8: Mostly accurate with minor inconsistencies
- 0.9-1.0: Highly accurate and factually sound"

# Hallucination Prompt
create_prompt "hallucination.txt" "Analyze the following text excerpt for potential hallucinations or fabricated information.

Content: {content}

Hallucination Detection Criteria:
- Identification of potentially non-existent organizations
- Detection of fabricated events or initiatives
- Verification of statistical claims
- Recognition of unsupported assertions

Specific Focus Areas:
- ABS-related organizations and initiatives
- Claims about biodiversity and genetic resources
- References to international protocols
- Purported research findings or project details

Evaluation Guidelines:
- Look for statements without verifiable sources
- Identify claims that contradict known scientific or policy information
- Detect overly specific or improbable assertions
- Check for coherence and plausibility of described scenarios

Respond in JSON format:
{
  \"hallucination_score\": 0.0-1.0,
  \"hallucination_confidence\": 0.0-1.0,
  \"issues\": [
    \"Specific hallucination or fabricated claim\",
    \"Unverifiable assertion\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: No significant hallucinations detected
- 0.3-0.5: Some unverifiable or suspicious claims
- 0.6-0.8: Multiple potentially fabricated statements
- 0.9-1.0: Extensive fabrication or hallucinations"

# Helpfulness Prompt
create_prompt "helpfulness.txt" "Evaluate the helpfulness of the following text excerpt for stakeholders in Access and Benefit Sharing (ABS).

Content: {content}

Helpfulness Assessment Criteria:
- Practical value of information
- Clarity of explanations
- Actionable insights
- Relevance to ABS stakeholders

Specific Evaluation Points:
- Does the content provide clear guidance?
- Are there actionable recommendations?
- Does it address key challenges in ABS?
- Is the information accessible to target audiences?

Dimensions of Helpfulness:
- Educational value
- Problem-solving potential
- Practical applicability
- Depth of insights
- Clarity of communication

Respond in JSON format:
{
  \"helpfulness_score\": 0.0-1.0,
  \"helpfulness_confidence\": 0.0-1.0,
  \"notes\": [
    \"Specific helpful aspect\",
    \"Potential improvement area\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: Minimal practical value
- 0.3-0.5: Some useful information
- 0.6-0.8: Substantially helpful
- 0.9-1.0: Extremely valuable and actionable"

# Relevance Prompt
create_prompt "relevance.txt" "Comprehensively evaluate the overall relevance of the following content to the ABS Initiative and broader Access and Benefit Sharing ecosystem.

Content: {content}

Relevance Assessment Criteria:
- Alignment with ABS principles
- Significance to current global biodiversity efforts
- Potential impact on policy or practice
- Depth of engagement with key ABS concepts

Specific Evaluation Dimensions:
- Current policy implications
- Innovative approaches to ABS
- Representation of stakeholder perspectives
- Contribution to scientific or policy discourse

Relevance Indicators:
- Explicit ABS discussions
- Genetic resources context
- Traditional knowledge references
- Biodiversity conservation connections
- Implementation challenges and solutions

Respond in JSON format:
{
  \"relevance_score\": 0.0-1.0,
  \"relevance_confidence\": 0.0-1.0,
  \"key_points\": [
    \"Specific relevance indicator\",
    \"Another key relevance factor\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: Minimal relevance to ABS
- 0.3-0.5: Tangential or peripheral connection
- 0.6-0.8: Substantive and meaningful relevance
- 0.9-1.0: Highly focused and directly critical to ABS"

# Toxicity Prompt
create_prompt "toxicity.txt" "Analyze the content for potentially harmful, offensive, or toxic language and perspectives.

Content: {content}

Toxicity Assessment Criteria:
- Presence of discriminatory language
- Harmful stereotypes or prejudices
- Inflammatory or aggressive tone
- Potential for causing harm or offense

Specific Evaluation Dimensions:
- Representation of diverse stakeholders
- Language used for different groups
- Potential for marginalizing perspectives
- Tone and implicit biases

Toxicity Indicators:
- Discriminatory statements
- Derogatory terminology
- Harmful stereotyping
- Dismissive or belittling language
- Inflammatory rhetoric

Respond in JSON format:
{
  \"toxicity_score\": 0.0-1.0,
  \"toxicity_confidence\": 0.0-1.0,
  \"concerns\": [
    \"Specific toxic language or perspective\",
    \"Potential harmful implication\"
  ]
}

Scoring Guidelines:
- 0.0-0.2: No toxic elements detected
- 0.3-0.5: Minor toxic undertones
- 0.6-0.8: Significant toxic language or perspectives
- 0.9-1.0: Highly offensive or harmful content"

# Summary Validation Prompt
create_prompt "summary_validation.txt" "Validate the accuracy and fidelity of the generated summary against the original content.

Summary: {summary}
Original Content: {content}

Validation Criteria:
- Factual accuracy of summary claims
- Preservation of original content's key messages
- Absence of hallucinations or fabricated information
- Maintenance of original context and nuance

Specific Evaluation Points:
- Do all summary statements exist in the original text?
- Are key points represented accurately?
- Has the original meaning been preserved?
- Are there any distortions or misrepresentations?

Validation Focus Areas:
- Factual statements
- Interpretative claims
- Contextual representation
- Potential information loss or addition

Respond in JSON format:
{
  \"valid\": true/false,
  \"issues\": [
    \"Specific validation concern\",
    \"Potential misrepresentation\"
  ],
  \"explanation\": \"Brief rationale for validation result\"
}

Validation Guidelines:
- Strict interpretation of content fidelity
- Careful cross-referencing of summary and original text
- Consideration of context and implied meanings
- Balanced assessment of summarization challenges"

echo "All specified prompts have been populated!"

# Make the script executable
chmod +x create_prompts.sh