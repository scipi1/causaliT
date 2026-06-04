import pandas as pd, json, os

base = 'experiments/2_ARCH_STUDY/BENCHMARKS/results'

# Check training metrics CSV columns
for folder in ['SWEEP_seed_bs_scm1c_586027', 'SWEEP_nares_Toe_CC_scm1c_585726', 'SWEEP_seed_single_Toe_CC_scm1c_1758745']:
    combos = os.path.join(base, folder, 'sweeper', 'runs', 'combinations')
    first_run = sorted(os.listdir(combos))[0]
    metrics_path = os.path.join(combos, first_run, 'k_0', 'logs', 'csv', 'version_0', 'metrics.csv')
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        print(f'{folder}:')
        print(f'  cols: {list(df.columns)}')
        print(f'  n_epochs: {df["epoch"].max() if "epoch" in df.columns else "no epoch col"}')
    else:
        print(f'{folder}: NO metrics.csv')
print()

# Check aggregate_dag.json block names
for folder in ['SWEEP_nares_Toe_CC_scm1c_585726', 'SWEEP_seed_single_Toe_CC_scm1c_1758745', 'SWEEP_seed_bs_scm1c_586027']:
    dag_path = os.path.join(base, folder, 'eval', 'eval_seed_sweep', 'files', 'aggregate_dag.json')
    if os.path.exists(dag_path):
        with open(dag_path) as f:
            d = json.load(f)
        print(f'{folder}:')
        print(f'  blocks: {list(d["blocks"].keys())}')
        print(f'  n_seeds: {d["n_seeds"]}')
    else:
        print(f'{folder}: NO aggregate_dag.json')
