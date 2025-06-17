-- Schema for storing training run metadata and metrics

-- Main training run table
CREATE TABLE training_runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    -- Basic run info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status ENUM('running', 'completed', 'failed', 'interrupted') NOT NULL,
    
    -- Timing
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    total_duration_seconds INT,
    last_checkpoint_time DATETIME,
    
    -- Static configuration (stored as JSON)
    config JSON NOT NULL,  -- Contains all TrainingConfig fields
    system_info JSON NOT NULL,  -- Contains CUDA device, GPU count, etc.
    dataset_info JSON NOT NULL,  -- Contains dataset stats, augmentation info
    
    -- Best checkpoint info
    best_checkpoint_path VARCHAR(255),
    best_validation_metric FLOAT,
    best_validation_metric_name VARCHAR(50),
    
    -- Current progress
    current_epoch INT NOT NULL DEFAULT 0,
    current_step INT NOT NULL DEFAULT 0,
    stop_reason VARCHAR(255),
    
    -- Training speed
    examples_per_second FLOAT,
    
    -- Metadata
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Table for storing metrics over time
CREATE TABLE training_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    step INT NOT NULL,
    epoch INT NOT NULL,
    
    -- Loss metrics
    text_loss FLOAT NOT NULL,
    grid_loss FLOAT NOT NULL,
    total_loss FLOAT NOT NULL,
    learning_rate FLOAT NOT NULL,
    
    -- Validation metrics (NULL if not a validation step)
    exact_match_accuracy FLOAT,
    avg_tile_difference FLOAT,
    
    -- Training speed
    examples_per_second FLOAT NOT NULL,
    
    -- Metadata
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id),
    INDEX idx_run_step (run_id, step)
);

-- Table for storing per-task validation metrics
CREATE TABLE task_validation_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    step INT NOT NULL,
    task_id VARCHAR(255) NOT NULL,  -- Reference to the original task
    
    -- Task-specific metrics
    exact_match BOOLEAN NOT NULL,
    tile_difference INT NOT NULL,
    text_loss FLOAT NOT NULL,
    grid_loss FLOAT NOT NULL,
    
    -- Metadata
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id),
    INDEX idx_run_task (run_id, task_id)
); 