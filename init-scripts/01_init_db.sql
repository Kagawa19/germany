-- ABS Initiative Metadata Database Schema
-- Comprehensive schema for tracking mentions of the ABS Initiative across various sources

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
    relevance_score FLOAT, -- how relevant/important is this mention (0.0 to 1.0)
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

-- SENTIMENT ANALYSIS TABLE
CREATE TABLE sentiment_analysis (
    source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE PRIMARY KEY,
    overall_sentiment VARCHAR(50), -- positive, negative, neutral
    sentiment_score FLOAT, -- -1.0 to 1.0
    sentiment_confidence FLOAT -- 0.0 to 1.0
);

-- VECTOR EMBEDDINGS TABLE (for semantic search)
-- Ensure the content_embeddings table exists before inserting data
CREATE TABLE IF NOT EXISTS content_embeddings (
    source_id INTEGER PRIMARY KEY REFERENCES content_sources(id) ON DELETE CASCADE,
    embedding_vector VECTOR(1536), -- Adjust the dimension if needed
    embedding_model VARCHAR(100)
);

-- Run this in your PostgreSQL terminal
CREATE OR REPLACE VIEW v_abs_content_analysis AS
SELECT 
    cs.id AS source_id,
    cs.url,
    cs.title,
    cs.publication_date,
    cs.language,
    cs.source_type,
    ARRAY_AGG(DISTINCT am.name_variant) AS name_variants,
    ARRAY_AGG(DISTINCT ta.theme) AS themes,
    ARRAY_AGG(DISTINCT gf.country) AS countries,
    ARRAY_AGG(DISTINCT gf.region) AS regions,
    ARRAY_AGG(DISTINCT o.name) AS organizations,
    ARRAY_AGG(DISTINCT r.resource_name) AS resources,
    ARRAY_AGG(DISTINCT aud.audience_type) AS target_audiences,
    sa.overall_sentiment,
    MAX(am.relevance_score) AS max_relevance_score,
    cs.content_summary
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
GROUP BY 
    cs.id, cs.url, cs.title, cs.publication_date, cs.language, 
    cs.source_type, sa.overall_sentiment, cs.content_summary;

-- Now proceed with inserting data
INSERT INTO content_embeddings (source_id, embedding_vector, embedding_model)
VALUES (%s, %s, %s)
ON CONFLICT (source_id) DO UPDATE 
SET embedding_vector = EXCLUDED.embedding_vector,
    embedding_model = EXCLUDED.embedding_model;


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

-- CREATE A TRIGGER TO AUTOMATICALLY UPDATE THE LAST_UPDATED_AT TIMESTAMP
CREATE OR REPLACE FUNCTION update_modified_column() RETURNS TRIGGER AS $$ 
BEGIN
    NEW.last_updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

CREATE TRIGGER update_content_timestamp
BEFORE UPDATE ON content_sources
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- CREATE VIEW FOR COMPREHENSIVE CONTENT ANALYSIS
CREATE OR REPLACE VIEW v_abs_content_analysis AS
SELECT 
    cs.id AS source_id,
    cs.url,
    cs.title,
    cs.publication_date,
    cs.language,
    cs.source_type,
    ARRAY_AGG(DISTINCT am.name_variant) AS name_variants,
    ARRAY_AGG(DISTINCT ta.theme) AS themes,
    ARRAY_AGG(DISTINCT gf.country) AS countries,
    ARRAY_AGG(DISTINCT gf.region) AS regions,
    ARRAY_AGG(DISTINCT o.name) AS organizations,
    ARRAY_AGG(DISTINCT r.resource_name) AS resources,
    ARRAY_AGG(DISTINCT aud.audience_type) AS target_audiences,
    sa.overall_sentiment,
    MAX(am.relevance_score) AS max_relevance_score,
    cs.content_summary
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
GROUP BY 
    cs.id, cs.url, cs.title, cs.publication_date, cs.language, 
    cs.source_type, sa.overall_sentiment, cs.content_summary;

-- CREATE FUNCTION TO SEARCH ACROSS NAME VARIANTS
CREATE OR REPLACE FUNCTION search_abs_by_name(search_term TEXT)
RETURNS TABLE (
    source_id INTEGER,
    url VARCHAR(255),
    title VARCHAR(255),
    publication_date DATE,
    name_variant VARCHAR(255),
    relevance_score FLOAT,
    mention_context TEXT
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
        am.mention_context
    FROM content_sources cs
    JOIN abs_mentions am ON cs.id = am.source_id
    WHERE 
        am.name_variant ILIKE '%' || search_term || '%'
    ORDER BY am.relevance_score DESC;
END;
$$ LANGUAGE plpgsql;