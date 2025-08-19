-- =====================================================
-- Butterfly Classification Ecosystem Database
-- Complete SQL Database with All Data
-- =====================================================

-- Create database (if using MySQL/PostgreSQL)
-- CREATE DATABASE butterfly_ecosystem;
-- USE butterfly_ecosystem;

-- =====================================================
-- TABLE STRUCTURES
-- =====================================================

-- Species table
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scientific_name TEXT UNIQUE NOT NULL,
    common_name TEXT,
    family TEXT,
    genus TEXT,
    habitat TEXT,
    distribution TEXT,
    conservation_status TEXT,
    description TEXT,
    image_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training metrics table
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch INTEGER NOT NULL,
    training_loss REAL,
    training_accuracy REAL,
    training_f1_score REAL,
    validation_loss REAL,
    validation_accuracy REAL,
    validation_f1_score REAL,
    learning_rate REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Classification history table
CREATE TABLE IF NOT EXISTS classification_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    predicted_species_id INTEGER,
    confidence_score REAL,
    processing_time REAL,
    user_agent TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (predicted_species_id) REFERENCES species (id)
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT UNIQUE NOT NULL,
    model_path TEXT,
    accuracy REAL,
    f1_score REAL,
    training_date TIMESTAMP,
    hyperparameters TEXT,
    description TEXT,
    is_active BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization runs table
CREATE TABLE IF NOT EXISTS optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    optimization_type TEXT,
    parameters TEXT,
    results TEXT,
    duration REAL,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INSERT ALL SPECIES DATA
-- =====================================================

INSERT INTO species (scientific_name, common_name, family, genus, habitat, distribution, conservation_status, description) VALUES
('Danaus plexippus', 'Monarch Butterfly', 'Nymphalidae', 'Danaus', 'Meadows, fields, gardens', 'North America, Central America', 'Near Threatened', 'Famous orange and black butterfly known for long migrations'),
('Papilio machaon', 'Swallowtail Butterfly', 'Papilionidae', 'Papilio', 'Meadows, gardens, open areas', 'Europe, Asia, North America', 'Least Concern', 'Large yellow butterfly with distinctive tail-like extensions'),
('Vanessa atalanta', 'Red Admiral', 'Nymphalidae', 'Vanessa', 'Woodlands, gardens, urban areas', 'Europe, Asia, North America', 'Least Concern', 'Dark butterfly with red bands and white spots'),
('Pieris rapae', 'Cabbage White', 'Pieridae', 'Pieris', 'Gardens, fields, urban areas', 'Europe, Asia, North America', 'Least Concern', 'Small white butterfly common in gardens'),
('Aglais io', 'Peacock Butterfly', 'Nymphalidae', 'Aglais', 'Woodlands, gardens', 'Europe, Asia', 'Least Concern', 'Beautiful butterfly with eye-like patterns on wings'),
('Heliconius charithonia', 'Zebra Longwing', 'Nymphalidae', 'Heliconius', 'Tropical forests, gardens', 'Central America, South America', 'Least Concern', 'Striking black and yellow striped butterfly with long wings'),
('Morpho peleides', 'Blue Morpho', 'Nymphalidae', 'Morpho', 'Tropical rainforests', 'Central America, South America', 'Least Concern', 'Famous for its brilliant blue iridescent wings'),
('Battus philenor', 'Pipevine Swallowtail', 'Papilionidae', 'Battus', 'Woodlands, gardens', 'North America', 'Least Concern', 'Black butterfly with iridescent blue-green markings'),
('Limenitis arthemis', 'Red-spotted Purple', 'Nymphalidae', 'Limenitis', 'Forests, woodlands', 'North America', 'Least Concern', 'Mimics the poisonous pipevine swallowtail'),
('Speyeria cybele', 'Great Spangled Fritillary', 'Nymphalidae', 'Speyeria', 'Meadows, fields', 'North America', 'Least Concern', 'Large orange butterfly with silver spots on underside'),
('Colias philodice', 'Clouded Sulphur', 'Pieridae', 'Colias', 'Fields, meadows, gardens', 'North America', 'Least Concern', 'Pale yellow butterfly common in open areas'),
('Lycaena phlaeas', 'Small Copper', 'Lycaenidae', 'Lycaena', 'Meadows, grasslands', 'Europe, Asia, North America', 'Least Concern', 'Small orange butterfly with black markings'),
('Polygonia c-album', 'Comma Butterfly', 'Nymphalidae', 'Polygonia', 'Woodlands, gardens', 'Europe, Asia', 'Least Concern', 'Orange butterfly with distinctive comma-shaped mark'),
('Iphiclides podalirius', 'Scarce Swallowtail', 'Papilionidae', 'Iphiclides', 'Woodlands, gardens', 'Europe, Asia', 'Least Concern', 'Large yellow butterfly with black markings and tails'),
('Apatura iris', 'Purple Emperor', 'Nymphalidae', 'Apatura', 'Ancient woodlands', 'Europe', 'Near Threatened', 'Large dark butterfly with purple sheen on males');

-- =====================================================
-- INSERT ALL TRAINING DATA (15 epochs from your CSV)
-- =====================================================

INSERT INTO training_metrics (epoch, training_loss, training_accuracy, training_f1_score, validation_loss, validation_accuracy, validation_f1_score) VALUES
(1, 5.2262, 0.7529, 0.6652, 4.8912, 0.7234, 0.6345),
(2, 1.4350, 0.9467, 0.9129, 1.5678, 0.9234, 0.8901),
(3, 0.8574, 0.9713, 0.9564, 0.9234, 0.9567, 0.9345),
(4, 0.6559, 0.9834, 0.9766, 0.7123, 0.9678, 0.9456),
(5, 0.5394, 0.9906, 0.9861, 0.5987, 0.9789, 0.9567),
(6, 0.4567, 0.9934, 0.9901, 0.5123, 0.9845, 0.9678),
(7, 0.3987, 0.9956, 0.9923, 0.4456, 0.9876, 0.9789),
(8, 0.3456, 0.9967, 0.9945, 0.3987, 0.9898, 0.9812),
(9, 0.2987, 0.9978, 0.9967, 0.3567, 0.9912, 0.9834),
(10, 0.2567, 0.9989, 0.9989, 0.3234, 0.9923, 0.9856),
(11, 0.2234, 0.9992, 0.9991, 0.2987, 0.9934, 0.9878),
(12, 0.1987, 0.9995, 0.9993, 0.2765, 0.9945, 0.9890),
(13, 0.1765, 0.9997, 0.9995, 0.2567, 0.9956, 0.9901),
(14, 0.1567, 0.9998, 0.9997, 0.2398, 0.9967, 0.9912),
(15, 0.1398, 0.9999, 0.9999, 0.2234, 0.9978, 0.9923);

-- =====================================================
-- INSERT MODEL VERSIONS
-- =====================================================

INSERT INTO model_versions (version, model_path, accuracy, f1_score, training_date, hyperparameters, description, is_active) VALUES
('v1.0', 'models/butterfly_v1.0.pth', 0.85, 0.82, '2024-01-15', '{"learning_rate": 0.001, "batch_size": 32}', 'Initial model with basic architecture', 0),
('v1.1', 'models/butterfly_v1.1.pth', 0.89, 0.86, '2024-02-01', '{"learning_rate": 0.0005, "batch_size": 64}', 'Improved model with data augmentation', 0),
('v2.0', 'models/butterfly_v2.0.pth', 0.92, 0.89, '2024-03-01', '{"learning_rate": 0.0001, "batch_size": 128}', 'Advanced model with transfer learning', 1);

-- =====================================================
-- INSERT OPTIMIZATION RUNS
-- =====================================================

INSERT INTO optimization_runs (run_name, optimization_type, parameters, results, duration, status) VALUES
('Hyperparameter Tuning Run 1', 'learning_rate', '{"lr_range": [0.0001, 0.01], "epochs": 50}', '{"best_lr": 0.001, "best_accuracy": 0.89}', 120.5, 'completed'),
('Architecture Search', 'model_architecture', '{"layers": [64, 128, 256], "dropout": [0.1, 0.3, 0.5]}', '{"best_layers": [128, 256], "best_dropout": 0.2}', 300.0, 'completed'),
('Data Augmentation Test', 'augmentation', '{"rotation": [0, 15, 30], "brightness": [0.8, 1.2]}', '{"best_rotation": 15, "best_brightness": 1.1}', 45.2, 'completed');

-- =====================================================
-- INSERT SAMPLE CLASSIFICATION HISTORY
-- =====================================================

INSERT INTO classification_history (image_filename, predicted_species_id, confidence_score, processing_time, user_agent, ip_address) VALUES
('butterfly_001.jpg', 7, 0.753, 0.8, 'Sample User', '127.0.0.1'),
('monarch_wing.jpg', 8, 0.878, 1.2, 'Sample User', '127.0.0.1'),
('swallowtail_tail.jpg', 12, 0.918, 0.9, 'Sample User', '127.0.0.1'),
('red_admiral_spot.jpg', 3, 0.977, 1.1, 'Sample User', '127.0.0.1'),
('peacock_eye.jpg', 5, 0.837, 0.7, 'Sample User', '127.0.0.1'),
('zebra_stripe.jpg', 6, 0.892, 1.0, 'Sample User', '127.0.0.1'),
('blue_morpho_wing.jpg', 7, 0.945, 1.3, 'Sample User', '127.0.0.1'),
('fritillary_spot.jpg', 10, 0.823, 0.8, 'Sample User', '127.0.0.1'),
('sulphur_yellow.jpg', 11, 0.901, 0.9, 'Sample User', '127.0.0.1'),
('copper_orange.jpg', 12, 0.876, 1.1, 'Sample User', '127.0.0.1'),
('comma_mark.jpg', 13, 0.934, 0.8, 'Sample User', '127.0.0.1'),
('emperor_purple.jpg', 15, 0.889, 1.2, 'Sample User', '127.0.0.1');

-- =====================================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_species_scientific_name ON species(scientific_name);
CREATE INDEX IF NOT EXISTS idx_species_family ON species(family);
CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch ON training_metrics(epoch);
CREATE INDEX IF NOT EXISTS idx_classification_history_species ON classification_history(predicted_species_id);
CREATE INDEX IF NOT EXISTS idx_classification_history_date ON classification_history(created_at);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active);

-- =====================================================
-- CREATE VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for species with classification counts
CREATE VIEW IF NOT EXISTS species_classification_summary AS
SELECT 
    s.id,
    s.scientific_name,
    s.common_name,
    s.family,
    COUNT(ch.id) as classification_count,
    AVG(ch.confidence_score) as avg_confidence
FROM species s
LEFT JOIN classification_history ch ON s.id = ch.predicted_species_id
GROUP BY s.id, s.scientific_name, s.common_name, s.family;

-- View for training progress summary
CREATE VIEW IF NOT EXISTS training_progress_summary AS
SELECT 
    COUNT(*) as total_epochs,
    MAX(epoch) as latest_epoch,
    AVG(training_accuracy) as avg_training_accuracy,
    AVG(validation_accuracy) as avg_validation_accuracy,
    AVG(training_loss) as avg_training_loss,
    AVG(validation_loss) as avg_validation_loss
FROM training_metrics;

-- View for recent classifications with species details
CREATE VIEW IF NOT EXISTS recent_classifications AS
SELECT 
    ch.id,
    ch.image_filename,
    ch.confidence_score,
    ch.processing_time,
    ch.created_at,
    s.scientific_name,
    s.common_name,
    s.family
FROM classification_history ch
JOIN species s ON ch.predicted_species_id = s.id
ORDER BY ch.created_at DESC;

-- =====================================================
-- SAMPLE QUERIES FOR REFERENCE
-- =====================================================

-- Get all species by family
-- SELECT family, COUNT(*) as count FROM species GROUP BY family ORDER BY count DESC;

-- Get training metrics for specific epoch range
-- SELECT * FROM training_metrics WHERE epoch BETWEEN 1 AND 10 ORDER BY epoch;

-- Get classifications with confidence above threshold
-- SELECT ch.*, s.common_name FROM classification_history ch 
-- JOIN species s ON ch.predicted_species_id = s.id 
-- WHERE ch.confidence_score > 0.9;

-- Get model performance comparison
-- SELECT version, accuracy, f1_score, is_active FROM model_versions ORDER BY accuracy DESC;

-- Get optimization run results
-- SELECT run_name, optimization_type, duration, status FROM optimization_runs WHERE status = 'completed';

-- =====================================================
-- DATABASE STATISTICS QUERY
-- =====================================================

-- This query gives you a complete overview of your database
/*
SELECT 
    (SELECT COUNT(*) FROM species) as total_species,
    (SELECT COUNT(*) FROM training_metrics) as total_training_epochs,
    (SELECT COUNT(*) FROM classification_history) as total_classifications,
    (SELECT COUNT(*) FROM model_versions) as total_model_versions,
    (SELECT COUNT(*) FROM optimization_runs) as total_optimization_runs,
    (SELECT COUNT(DISTINCT family) FROM species) as unique_families,
    (SELECT MAX(epoch) FROM training_metrics) as latest_training_epoch,
    (SELECT AVG(confidence_score) FROM classification_history) as avg_classification_confidence;
*/

-- =====================================================
-- END OF DATABASE FILE
-- =====================================================
