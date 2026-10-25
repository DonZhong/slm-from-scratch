
BEGIN;

-- 硬件监控历史快照表
CREATE TABLE hardware_snapshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- 可选关联训练实验，便于分析训练期间的硬件表现
    training_run_id BIGINT
        REFERENCES training_runs(id)
        ON DELETE SET NULL,

    -- 硬件数据实际采样时间
    sampled_at TIMESTAMPTZ NOT NULL,

    -- 采集数据的主机名称
    host_name TEXT NOT NULL
        CHECK (length(trim(host_name)) > 0),

    -- CPU、内存、磁盘、网络、GPU 等指标
    metrics JSONB NOT NULL
        CHECK (jsonb_typeof(metrics) = 'object'),

    -- 数据进入数据库的时间
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 支持按时间倒序查询最近的硬件快照
CREATE INDEX idx_hardware_snapshots_sampled_at
    ON hardware_snapshots (sampled_at DESC, id DESC);

-- 支持查看某一次训练期间的硬件变化
CREATE INDEX idx_hardware_snapshots_training_run
    ON hardware_snapshots (training_run_id, sampled_at DESC)
    WHERE training_run_id IS NOT NULL;

COMMIT;
