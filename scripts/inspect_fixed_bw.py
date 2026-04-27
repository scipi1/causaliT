import pandas as pd

paths = {
    'ORC_HSIC':  'experiments/tests/hsic_noise_floor/results/oracle_hsic_fixed_bw_64831559/k_0/logs/csv/version_0/metrics.csv',
    'ANT_HSIC':  'experiments/tests/hsic_noise_floor/results/anti_oracle_hsic_fixed_bw_64831575/k_0/logs/csv/version_0/metrics.csv',
    'ORC_NHSIC': 'experiments/tests/hsic_noise_floor/results/oracle_nhsic_fixed_bw_64831552/k_0/logs/csv/version_0/metrics.csv',
    'ANT_NHSIC': 'experiments/tests/hsic_noise_floor/results/anti_oracle_nhsic_fixed_bw_64831564/k_0/logs/csv/version_0/metrics.csv',
}
for k, v in paths.items():
    raw = pd.read_csv(v)
    df = raw.groupby('epoch', as_index=False).first().sort_values('epoch').reset_index(drop=True)
    print(f'=== {k} (epochs_logged={int(df.epoch.max()+1)}) ===')
    cols = ['val_hsic_cross', 'val_hsic_self', 'val_x_rmse', 'val_x_r2', 'train_hsic_cross', 'train_hsic_self', 'train_x_rmse']
    cols = [c for c in cols if c in df.columns]
    n = len(df)
    idxs = sorted({0, 1, 2, 5, 10, 20, 40, 80, max(0, n//4), max(0, n//2), max(0, 3*n//4), n-1})
    idxs = [i for i in idxs if i < n]
    print(df[['epoch'] + cols].iloc[idxs].to_string(index=False))
    # last-50 means
    last = df.tail(50)
    print('  last50_mean:', {c: round(float(last[c].mean()), 5) for c in cols})
    print()
