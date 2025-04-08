-- ABS Initiative Metadata Database Schema Initialization

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- MAIN CONTENT TABLE
CREATE TABLE content_sources (
    id SERIAL PRIMARY KEY,
    url VARCHAR(255) NOT NULL UNIQUE,
    domain_name VARCHAR(100),
    title VARCHAR(255),
    publication_date DATE,
    source_type VARCHAR(50), -- official document, news article, blog, academic paper, etc.
    language VARCHAR(50),
    full_content TEXT,
    content_summary TEXT,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- MENTIONS TABLE (links to content_sources)
CREATE TABLE abs_mentions (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    name_variant VARCHAR(255), -- exact name variant used (ABS CDI, Initiative APA, etc.)
    mention_context TEXT, -- paragraph or sentence containing the reference
    mention_type VARCHAR(100), -- passing reference, detailed description, project partner, etc.
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0.0 AND 1.0), -- how relevant/important is this mention
    mention_position INTEGER, -- position in the document (paragraph number)
    UNIQUE(source_id, mention_position)
);

-- GEOGRAPHIC FOCUS TABLE
CREATE TABLE geographic_focus (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    mention_id INTEGER REFERENCES abs_mentions(id) ON DELETE CASCADE,
    country VARCHAR(100),
    region VARCHAR(100),
    scope VARCHAR(50) -- national, regional, continental, global
);

-- THEMATIC AREAS TABLE
CREATE TABLE thematic_areas (
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    theme VARCHAR(100), -- biodiversity, indigenous knowledge, genetic resources, etc.
    PRIMARY KEY (source_id, theme)
);

-- PROJECT DETAILS TABLE
CREATE TABLE project_details (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    project_name VARCHAR(255),
    project_type VARCHAR(100), -- workshop, training, policy development, etc.
    start_date DATE,
    end_date DATE,
    status VARCHAR(50), -- planned, ongoing, completed
    description TEXT
);

-- ORGANIZATIONS TABLE
CREATE TABLE organizations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    organization_type VARCHAR(100), -- partner, funder, beneficiary, implementer
    website VARCHAR(255)
);

-- ORGANIZATION MENTIONS JUNCTION TABLE
CREATE TABLE organization_mentions (
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100), -- partner, funder, beneficiary, implementer
    description TEXT,
    PRIMARY KEY (source_id, organization_id)
);

-- RESOURCES TABLE
CREATE TABLE resources (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    resource_type VARCHAR(100), -- publication, tool, case study, etc.
    resource_name VARCHAR(255),
    resource_url VARCHAR(255),
    description TEXT
);

-- TARGET AUDIENCES TABLE
CREATE TABLE target_audiences (
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
    audience_type VARCHAR(100), -- policymakers, indigenous communities, researchers, etc.
    PRIMARY KEY (source_id, audience_type)
);

-- SENTIMENT ANALYSIS TABLE (Enhanced with robust constraints)
CREATE TABLE sentiment_analysis (
    source_id INTEGER PRIMARY KEY REFERENCES content_sources(id) ON DELETE CASCADE,
    overall_sentiment VARCHAR(50) DEFAULT 'Neutral', -- positive, negative, neutral
    sentiment_score NUMERIC(5,2) DEFAULT 0.0 CHECK (sentiment_score BETWEEN -1.0 AND 1.0), -- -1.0 to 1.0
    sentiment_confidence NUMERIC(5,2) DEFAULT 0.5 CHECK (sentiment_confidence BETWEEN 0.0 AND 1.0) -- 0.0 to 1.0
);

-- ENHANCED CONTENT SCORES TABLE (NEW)
CREATE TABLE content_scores (
    source_id INTEGER PRIMARY KEY REFERENCES content_sources(id) ON DELETE CASCADE,
    context_relevance_score NUMERIC(5,2) DEFAULT 0.0 CHECK (context_relevance_score BETWEEN 0.0 AND 1.0),
    context_relevance_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (context_relevance_confidence BETWEEN 0.0 AND 1.0),
    conciseness_score NUMERIC(5,2) DEFAULT 0.0 CHECK (conciseness_score BETWEEN 0.0 AND 1.0),
    conciseness_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (conciseness_confidence BETWEEN 0.0 AND 1.0),
    correctness_score NUMERIC(5,2) DEFAULT 0.0 CHECK (correctness_score BETWEEN 0.0 AND 1.0),
    correctness_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (correctness_confidence BETWEEN 0.0 AND 1.0),
    hallucination_score NUMERIC(5,2) DEFAULT 0.0 CHECK (hallucination_score BETWEEN 0.0 AND 1.0),
    hallucination_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (hallucination_confidence BETWEEN 0.0 AND 1.0),
    helpfulness_score NUMERIC(5,2) DEFAULT 0.0 CHECK (helpfulness_score BETWEEN 0.0 AND 1.0),
    helpfulness_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (helpfulness_confidence BETWEEN 0.0 AND 1.0),
    relevance_score NUMERIC(5,2) DEFAULT 0.0 CHECK (relevance_score BETWEEN 0.0 AND 1.0),
    relevance_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (relevance_confidence BETWEEN 0.0 AND 1.0),
    toxicity_score NUMERIC(5,2) DEFAULT 0.0 CHECK (toxicity_score BETWEEN 0.0 AND 1.0),
    toxicity_confidence NUMERIC(5,2) DEFAULT 0.0 CHECK (toxicity_confidence BETWEEN 0.0 AND 1.0),
    issues_json JSONB -- store detected issues in JSON format
);

-- VECTOR EMBEDDINGS TABLE (Enhanced to support both vector and JSON)
CREATE TABLE content_embeddings (
    source_id INTEGER PRIMARY KEY REFERENCES content_sources(id) ON DELETE CASCADE,
    embedding_vector VECTOR(1536) NULL,  -- For pgvector support
    embedding_json JSONB NULL,  -- Fallback JSON storage
    embedding_model VARCHAR(100)
);

-- CREATE INDEXES FOR FASTER QUERIES
CREATE INDEX idx_content_sources_domain ON content_sources(domain_name);
CREATE INDEX idx_content_sources_date ON content_sources(publication_date);
CREATE INDEX idx_content_sources_language ON content_sources(language);
CREATE INDEX idx_content_sources_type ON content_sources(source_type);

CREATE INDEX idx_abs_mentions_variant ON abs_mentions(name_variant);
CREATE INDEX idx_abs_mentions_type ON abs_mentions(mention_type);
CREATE INDEX idx_abs_mentions_relevance ON abs_mentions(relevance_score);

CREATE INDEX idx_geographic_focus_country ON geographic_focus(country);
CREATE INDEX idx_geographic_focus_region ON geographic_focus(region);

CREATE INDEX idx_organization_mentions_relation ON organization_mentions(relationship_type);

CREATE INDEX idx_resources_type ON resources(resource_type);

CREATE INDEX idx_sentiment_overall ON sentiment_analysis(overall_sentiment);
CREATE INDEX idx_content_embeddings_model ON content_embeddings(embedding_model);

-- NEW INDEXES FOR CONTENT SCORES
CREATE INDEX idx_content_scores_relevance ON content_scores(relevance_score);
CREATE INDEX idx_content_scores_correctness ON content_scores(correctness_score);
CREATE INDEX idx_content_scores_toxicity ON content_scores(toxicity_score);

-- FUNCTION TO AUTOMATICALLY UPDATE LAST_UPDATED_AT TIMESTAMP
CREATE OR REPLACE FUNCTION update_modified_column() 
RETURNS TRIGGER AS $$ 
BEGIN
    NEW.last_updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- TRIGGER TO UPDATE TIMESTAMP
CREATE TRIGGER update_content_timestamp
BEFORE UPDATE ON content_sources
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- FUNCTION TO VALIDATE SENTIMENT SCORES
CREATE OR REPLACE FUNCTION validate_sentiment_score()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure sentiment score is between -1.0 and 1.0
    NEW.sentiment_score := GREATEST(-1.0, LEAST(NEW.sentiment_score, 1.0));
    
    -- Ensure sentiment confidence is between 0.0 and 1.0
    NEW.sentiment_confidence := GREATEST(0.0, LEAST(NEW.sentiment_confidence, 1.0));
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- TRIGGER TO ENFORCE SENTIMENT CONSTRAINTS
CREATE TRIGGER enforce_sentiment_constraints
BEFORE INSERT OR UPDATE ON sentiment_analysis
FOR EACH ROW
EXECUTE FUNCTION validate_sentiment_score();

-- FUNCTION TO VALIDATE CONTENT SCORES
CREATE OR REPLACE FUNCTION validate_content_scores()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure all scores are between 0.0 and 1.0
    NEW.context_relevance_score := GREATEST(0.0, LEAST(NEW.context_relevance_score, 1.0));
    NEW.context_relevance_confidence := GREATEST(0.0, LEAST(NEW.context_relevance_confidence, 1.0));
    NEW.conciseness_score := GREATEST(0.0, LEAST(NEW.conciseness_score, 1.0));
    NEW.conciseness_confidence := GREATEST(0.0, LEAST(NEW.conciseness_confidence, 1.0));
    NEW.correctness_score := GREATEST(0.0, LEAST(NEW.correctness_score, 1.0));
    NEW.correctness_confidence := GREATEST(0.0, LEAST(NEW.correctness_confidence, 1.0));
    NEW.hallucination_score := GREATEST(0.0, LEAST(NEW.hallucination_score, 1.0));
    NEW.hallucination_confidence := GREATEST(0.0, LEAST(NEW.hallucination_confidence, 1.0));
    NEW.helpfulness_score := GREATEST(0.0, LEAST(NEW.helpfulness_score, 1.0));
    NEW.helpfulness_confidence := GREATEST(0.0, LEAST(NEW.helpfulness_confidence, 1.0));
    NEW.relevance_score := GREATEST(0.0, LEAST(NEW.relevance_score, 1.0));
    NEW.relevance_confidence := GREATEST(0.0, LEAST(NEW.relevance_confidence, 1.0));
    NEW.toxicity_score := GREATEST(0.0, LEAST(NEW.toxicity_score, 1.0));
    NEW.toxicity_confidence := GREATEST(0.0, LEAST(NEW.toxicity_confidence, 1.0));
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- TRIGGER TO ENFORCE CONTENT SCORE CONSTRAINTS
CREATE TRIGGER enforce_content_score_constraints
BEFORE INSERT OR UPDATE ON content_scores
FOR EACH ROW
EXECUTE FUNCTION validate_content_scores();

-- COMPREHENSIVE CONTENT ANALYSIS VIEW (UPDATED)
CREATE OR REPLACE VIEW v_abs_content_analysis AS
SELECT 
    cs.id AS source_id,
    cs.url,
    cs.domain_name,
    cs.title,
    cs.publication_date,
    cs.language,
    cs.source_type,
    cs.content_summary,
    cs.full_content,
    ARRAY_AGG(DISTINCT am.name_variant) FILTER (WHERE am.name_variant IS NOT NULL) AS name_variants,
    ARRAY_AGG(DISTINCT ta.theme) FILTER (WHERE ta.theme IS NOT NULL) AS themes,
    ARRAY_AGG(DISTINCT gf.country) FILTER (WHERE gf.country IS NOT NULL) AS countries,
    ARRAY_AGG(DISTINCT gf.region) FILTER (WHERE gf.region IS NOT NULL) AS regions,
    ARRAY_AGG(DISTINCT o.name) FILTER (WHERE o.name IS NOT NULL) AS organizations,
    ARRAY_AGG(DISTINCT r.resource_name) FILTER (WHERE r.resource_name IS NOT NULL) AS resources,
    ARRAY_AGG(DISTINCT aud.audience_type) FILTER (WHERE aud.audience_type IS NOT NULL) AS target_audiences,
    sa.overall_sentiment,
    sa.sentiment_score,
    sa.sentiment_confidence,
    sc.context_relevance_score,
    sc.context_relevance_confidence,
    sc.conciseness_score,
    sc.conciseness_confidence,
    sc.correctness_score,
    sc.correctness_confidence,
    sc.hallucination_score,
    sc.hallucination_confidence,
    sc.helpfulness_score,
    sc.helpfulness_confidence,
    sc.relevance_score,
    sc.relevance_confidence,
    sc.toxicity_score,
    sc.toxicity_confidence,
    sc.issues_json,
    MAX(am.relevance_score) AS max_relevance_score,
    cs.extracted_at,
    cs.last_updated_at
FROM 
    content_sources cs
LEFT JOIN abs_mentions am ON cs.id = am.source_id
LEFT JOIN thematic_areas ta ON cs.id = ta.source_id
LEFT JOIN geographic_focus gf ON cs.id = gf.source_id
LEFT JOIN organization_mentions om ON cs.id = om.source_id
LEFT JOIN organizations o ON om.organization_id = o.id
LEFT JOIN resources r ON cs.id = r.source_id
LEFT JOIN target_audiences aud ON cs.id = aud.source_id
LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
LEFT JOIN content_scores sc ON cs.id = sc.source_id
GROUP BY 
    cs.id, cs.url, cs.domain_name, cs.title, cs.publication_date, cs.language, 
    cs.source_type, cs.content_summary, cs.full_content, cs.extracted_at, cs.last_updated_at,
    sa.overall_sentiment, sa.sentiment_score, sa.sentiment_confidence,
    sc.context_relevance_score, sc.context_relevance_confidence,
    sc.conciseness_score, sc.conciseness_confidence,
    sc.correctness_score, sc.correctness_confidence,
    sc.hallucination_score, sc.hallucination_confidence,
    sc.helpfulness_score, sc.helpfulness_confidence,
    sc.relevance_score, sc.relevance_confidence,
    sc.toxicity_score, sc.toxicity_confidence, sc.issues_json;

-- FUNCTION TO SEARCH ACROSS NAME VARIANTS
CREATE OR REPLACE FUNCTION search_abs_by_name(search_term TEXT)
RETURNS TABLE (
    source_id INTEGER,
    url VARCHAR(255),
    title VARCHAR(255),
    publication_date DATE,
    name_variant VARCHAR(255),
    relevance_score FLOAT,
    mention_context TEXT,
    overall_sentiment VARCHAR(50),
    correctness_score NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.id AS source_id,
        cs.url,
        cs.title,
        cs.publication_date,
        am.name_variant,
        am.relevance_score,
        am.mention_context,
        sa.overall_sentiment,
        sc.correctness_score
    FROM content_sources cs
    JOIN abs_mentions am ON cs.id = am.source_id
    LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
    LEFT JOIN content_scores sc ON cs.id = sc.source_id
    WHERE 
        am.name_variant ILIKE '%' || search_term || '%'
    ORDER BY am.relevance_score DESC;
END;
$$ LANGUAGE plpgsql;