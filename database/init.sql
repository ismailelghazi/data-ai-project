-- Initialize database schema for emotion detection predictions

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    emotion VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on created_at for faster queries
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

-- Create index on emotion for filtering
CREATE INDEX IF NOT EXISTS idx_predictions_emotion ON predictions(emotion);

-- Insert a sample record (optional)
-- INSERT INTO predictions (emotion, confidence) VALUES ('Happy', 0.95);
