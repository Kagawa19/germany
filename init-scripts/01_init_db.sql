-- Create the table with the specified columns
CREATE TABLE IF NOT EXISTS content_data (
    id SERIAL PRIMARY KEY,
    link VARCHAR(255) NOT NULL,
    summary TEXT,
    full_content TEXT,
    information TEXT,
    theme VARCHAR(100),
    organization VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_content_data_link ON content_data (link);
CREATE INDEX IF NOT EXISTS idx_content_data_theme ON content_data (theme);
CREATE INDEX IF NOT EXISTS idx_content_data_organization ON content_data (organization);

-- Insert some sample data
INSERT INTO content_data (link, summary, full_content, information, theme, organization)
VALUES 
    ('https://example.com/article1', 
     'Example article summary', 
     'This is the full content of the article. It contains detailed information about the topic.', 
     'Additional information about the article', 
     'Technology', 
     'Example Org'),
     
    ('https://example.com/article2', 
     'Another example article', 
     'Full content for the second article with more detailed information.', 
     'More metadata about this article', 
     'Science', 
     'Research Institute');