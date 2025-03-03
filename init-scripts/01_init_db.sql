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