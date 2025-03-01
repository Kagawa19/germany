-- Create the content_data table with the specified columns
CREATE TABLE IF NOT EXISTS content_data (
    id SERIAL PRIMARY KEY,
    link VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    date DATE,
    summary TEXT,
    full_content TEXT,
    information TEXT,
    theme VARCHAR(100),
    organization VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries on content_data
CREATE INDEX IF NOT EXISTS idx_content_data_link ON content_data (link);
CREATE INDEX IF NOT EXISTS idx_content_data_theme ON content_data (theme);
CREATE INDEX IF NOT EXISTS idx_content_data_organization ON content_data (organization);
CREATE INDEX IF NOT EXISTS idx_content_data_date ON content_data (date);

-- Create the benefits table
CREATE TABLE IF NOT EXISTS benefits (
    id SERIAL PRIMARY KEY,
    links TEXT[], -- Array to store multiple links
    benefits_to_germany TEXT,
    insights TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance on benefits
CREATE INDEX IF NOT EXISTS idx_benefits_links ON benefits USING GIN(links);

-- Create a relationship table (for many-to-many relationships)
CREATE TABLE IF NOT EXISTS content_benefits (
    content_id INTEGER REFERENCES content_data(id) ON DELETE CASCADE,
    benefit_id INTEGER REFERENCES benefits(id) ON DELETE CASCADE,
    PRIMARY KEY (content_id, benefit_id)
);