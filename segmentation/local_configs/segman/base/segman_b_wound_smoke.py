_base_ = [
    './segman_b_wound.py'
]

# Smoke test overrides: run one epoch with tiny batch/workers to catch errors quickly
data = dict(samples_per_gpu=2, workers_per_gpu=1)

# Use one-epoch runner to make the run very short (delete base runner first)
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=1)

# Reduce evaluation frequency so it runs once at end
evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'])
