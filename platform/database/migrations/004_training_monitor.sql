BEGIN;
ALTER TABLE training_runs ADD COLUMN runtime JSONB NOT NULL DEFAULT '{}'::jsonb
    CHECK (jsonb_typeof(runtime) = 'object');
ALTER TABLE training_runs ADD COLUMN monitor_updated_at TIMESTAMPTZ;
CREATE INDEX training_metrics_diagnostics_step_idx ON training_metrics (run_id, global_step DESC)
    WHERE diagnostics IS NOT NULL;
COMMIT;
