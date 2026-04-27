"""Inspect last-50-epoch summary across the noise-floor runs."""
import os
import sys
import pandas as pd

paths = {
    'B2048_ORC_HSIC_adapt'    : 'experiments/tests/hsic_noise_floor/results/noise_floor_hsic_bias_64807812/k_0/logs/csv/version_0/metrics.csv',
    'B2048_ORC_NHSIC_adapt'   : 'experiments/tests/hsic_noise_floor/results/noise_floor_hsic_norm_64807799/k_0/logs/csv/version_0/metrics.csv',
    'B2048_ANT_NHSIC_adapt'   : 'experiments/tests/hsic_noise_floor/results/noise_floor_hsic_norm_random_oracle_64813818/k_0/logs/csv/version_0/metrics.csv',
    'B2048_ANT_HSIC_adapt'    : 'experiments/tests/hsic_noise_floor/results/noise_floor_random_oracle_64807789/k_0/logs/csv/version_0/metrics.csv',
    'B128_ORC_HSIC_adapt'     : 'experiments/tests/hsic_noise_floor/results/noise_floor_small_batch_64807781/k_0/logs/csv/version_0/metrics.csv',
    'B128_ANT_HSIC_adapt'     : 'experiments/tests/hsic_noise_floor/results/noise_floor_small_batch_anti_64830339/k_0/logs/csv/version_0/metrics.csv',
}

for k, v in paths.items():
    if not os.path.exists(v):
        print(f'MISSING: {k} -> {v}')
        continue
    raw = pd.read_csv(v)
    df  = raw.groupby('epoch', as_index=False).first().sort_values('epoch').reset_index(drop=True)
    last = df.tail(50)
    cols = ['val_hsic_cross', 'val_hsic_self', 'val_x_rmse', 'val_x_r2']
    cols = [c for c in cols if c in last.columns]
    summary = {c: round(float(last[c].mean()), 5) for c in cols}
    print(f'{k:30s} ep={int(df.epoch.max()+1):4d}  {summary}')
