BEGIN;

-- Optional snapshots; existing experiments and metrics remain valid.
ALTER TABLE training_metrics
    ADD COLUMN diagnostics JSONB
    CHECK (diagnostics IS NULL OR jsonb_typeof(diagnostics) = 'object');

COMMIT;
