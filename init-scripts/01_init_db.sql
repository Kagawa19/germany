-- Create the content_data table with all columns in one place
CREATE TABLE IF NOT EXISTS content_data (
    id SERIAL PRIMARY KEY,
    link VARCHAR(255) NOT NULL UNIQUE,
    title VARCHAR(255),
    date DATE,
    summary TEXT,
    full_content TEXT,
    information TEXT,
    themes TEXT[],
    organization VARCHAR(100),
    sentiment VARCHAR(50),
    benefits_to_germany TEXT,
    insights TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries on content_data
CREATE INDEX IF NOT EXISTS idx_content_data_link ON content_data (link);
CREATE INDEX IF NOT EXISTS idx_content_data_themes ON content_data USING GIN (themes);
CREATE INDEX IF NOT EXISTS idx_content_data_organization ON content_data (organization);
CREATE INDEX IF NOT EXISTS idx_content_data_date ON content_data (date);
CREATE INDEX IF NOT EXISTS idx_content_data_sentiment ON content_data (sentiment);

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

CREATE TRIGGER update_content_timestamp
BEFORE UPDATE ON content_data
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- Add new columns to the content_data table if they don't exist
DO $$
BEGIN
    -- Check if initiative column exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='content_data' AND column_name='initiative') THEN
        ALTER TABLE content_data ADD COLUMN initiative VARCHAR(100);
    END IF;

    -- Check if initiative_key column exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='content_data' AND column_name='initiative_key') THEN
        ALTER TABLE content_data ADD COLUMN initiative_key VARCHAR(50);
    END IF;

    -- Check if benefit_categories column exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='content_data' AND column_name='benefit_categories') THEN
        ALTER TABLE content_data ADD COLUMN benefit_categories JSONB;
    END IF;

    -- Check if benefit_examples column exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='content_data' AND column_name='benefit_examples') THEN
        ALTER TABLE content_data ADD COLUMN benefit_examples JSONB;
    END IF;
END
$$;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_content_data_initiative ON content_data (initiative);
CREATE INDEX IF NOT EXISTS idx_content_data_initiative_key ON content_data (initiative_key);

-- Update the content_data table with initiative values for existing records (optional)
-- This assumes you want to classify existing records
UPDATE content_data
SET initiative = 'ABS Capacity Development Initiative',
    initiative_key = 'abs_cdi'
WHERE (
    full_content ILIKE '%ABS Capacity Development Initiative%' OR
    full_content ILIKE '%ABS CDI%' OR
    full_content ILIKE '%Access and Benefit Sharing%'
) AND initiative IS NULL;

UPDATE content_data
SET initiative = 'Bio-innovation Africa',
    initiative_key = 'bio_innovation_africa'
WHERE (
    full_content ILIKE '%Bio-innovation Africa%' OR
    full_content ILIKE '%BioInnovation Africa%'
) AND initiative IS NULL;