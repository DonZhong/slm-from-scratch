
BEGIN;

-- 1. 训练实验表
CREATE TABLE training_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'TinyGPT',
    dataset_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created'
        CHECK (
            status IN (
                'created',
                'running',
                'completed',
                'failed',
                'cancelled'
            )
        ),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- 2. 训练指标表
CREATE TABLE training_metrics (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(id) ON DELETE RESTRICT,
    global_step BIGINT NOT NULL CHECK (global_step > 0),
    epoch INTEGER NOT NULL CHECK (epoch > 0),
    train_loss DOUBLE PRECISION NOT NULL
        CHECK (train_loss >= 0),
    val_loss DOUBLE PRECISION
        CHECK (val_loss >= 0),
    learning_rate DOUBLE PRECISION NOT NULL
        CHECK (learning_rate > 0),
    tokens_per_second DOUBLE PRECISION
        CHECK (tokens_per_second >= 0),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, global_step)
);

-- 3. Checkpoint 元数据表
CREATE TABLE checkpoints (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(id) ON DELETE RESTRICT,
    global_step BIGINT NOT NULL CHECK (global_step >= 0),
    file_path TEXT NOT NULL UNIQUE,
    file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
    saved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, global_step)
);

COMMIT;
