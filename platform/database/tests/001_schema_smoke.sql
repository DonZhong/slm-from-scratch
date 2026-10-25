
\set ON_ERROR_STOP on

BEGIN;

-- 1. 创建一条模拟训练实验
INSERT INTO training_runs (
    name,
    model_name,
    dataset_name,
    status,
    config
)
VALUES (
    'schema-smoke-run',
    'TinyGPT',
    'Tiny Shakespeare',
    'completed',
    '{"batch_size": 4, "learning_rate": 0.001}'::jsonb
)
RETURNING id AS run_id
\gset

-- 2. 写入一条训练指标
INSERT INTO training_metrics (
    run_id,
    global_step,
    epoch,
    train_loss,
    val_loss,
    learning_rate,
    tokens_per_second
)
VALUES (
    :run_id,
    10,
    1,
    3.618338,
    3.269403,
    0.001,
    1200.0
);

-- 3. 写入模拟 Checkpoint 元数据
INSERT INTO checkpoints (
    run_id,
    global_step,
    file_path,
    file_size_bytes
)
VALUES (
    :run_id,
    10,
    '/tmp/slm-schema-smoke-step-10.pt',
    102400
);

-- 4. 通过外键关联三张表
SELECT
    r.id AS run_id,
    r.name AS run_name,
    m.global_step,
    m.train_loss,
    m.val_loss,
    c.file_path
FROM training_runs AS r
JOIN training_metrics AS m
    ON m.run_id = r.id
JOIN checkpoints AS c
    ON c.run_id = r.id
WHERE r.id = :run_id;

-- 5. 撤销测试数据
ROLLBACK;

-- 6. 验证模拟实验没有被保留
SELECT COUNT(*) AS remaining_test_runs
FROM training_runs
WHERE name = 'schema-smoke-run';
