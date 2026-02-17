# Parallel Sweep Usage Guide

This guide explains how to use the new parallel sweep functionality that runs parameter combinations in parallel using SLURM job arrays.

## Overview

The parallel sweep system allows you to run parameter sweeps in parallel on cluster environments, significantly reducing the time needed to complete large parameter searches. It uses SLURM job arrays and follows the scratch folder pattern for efficient resource usage.

## Key Features

- **Backward Compatible**: Existing code using `@combination_sweep` continues to work unchanged
- **Cluster Optimized**: Uses SLURM job arrays with scratch folder management
- **Resource Aware**: Respects cluster limits (default: max 6 concurrent jobs)
- **Fault Tolerant**: Individual job failures don't affect other combinations
- **Progress Tracking**: Easy monitoring of sweep progress

## Usage

### Method 1: Using the Shell Script (Recommended)

The easiest way to run parallel sweeps is using the provided shell script:

1. **Copy and modify the script:**
   ```bash
   cp teacher_student/scripts/train_parallel_sweep.sh my_experiment_sweep.sh
   ```

2. **Edit the configuration variables:**
   ```bash
   # Edit these variables in the script
   EXPERIMENT_ID="my_experiment"
   MAX_CONCURRENT_JOBS=6
   WALLTIME="5-00:00:00"
   GPU_MEM="11g"
   MEM_PER_CPU="10g"
   ```

3. **Submit to SLURM:**
   ```bash
   sbatch my_experiment_sweep.sh
   ```

### Method 2: Direct CLI Usage

You can also run parallel sweeps directly with the CLI:

```bash
# Sequential sweep (existing behavior)
python teacher_student/cli.py train --exp_id my_experiment --cluster True

# Parallel sweep (new functionality)
python teacher_student/cli.py train --exp_id my_experiment --cluster True --parallel True
```

### Advanced Options

You can customize the parallel execution with additional parameters:

```bash
python teacher_student/cli.py train \
    --exp_id my_experiment \
    --cluster True \
    --parallel True \
    --max_concurrent_jobs 8 \
    --walltime "10-00:00:00" \
    --gpu_mem "48g" \
    --mem_per_cpu "16g"
```

### Available Parameters

- `--parallel`: Enable parallel execution (default: False)
- `--max_concurrent_jobs`: Maximum concurrent SLURM jobs (default: 6)
- `--walltime`: SLURM walltime limit (default: "5-00:00:00")
- `--gpu_mem`: GPU memory requirement (default: "11g")
- `--mem_per_cpu`: CPU memory requirement (default: "10g")
- `--submit_jobs`: Actually submit jobs vs. dry run (default: True)

## How It Works

1. **Preparation Phase**: 
   - Reads your config and sweep files
   - Generates all parameter combinations
   - Creates a `combinations_data.json` file with all combinations
   - Generates a SLURM job array script (`run_sweep_array.sh`)

2. **Execution Phase**:
   - Submits the job array to SLURM
   - Each array job runs one parameter combination
   - Jobs use scratch folders for large result files
   - Results are stored in the scratch location during execution

3. **Monitoring**:
   - Job ID is saved to `job_id.txt` for reference
   - SLURM output files show individual job progress
   - Use standard SLURM commands (`squeue`, `sacct`) to monitor

## File Structure

When using parallel sweeps, the following files are created:

```
experiments/training/my_experiment/
├── config.yaml                    # Original config
├── sweep.yaml                     # Original sweep config
├── combinations_data.json         # Generated combinations data
├── run_sweep_array.sh            # Generated SLURM script
├── job_id.txt                    # SLURM job ID
├── slurm_logs/                   # SLURM output logs
│   ├── sweep_output_12345_0.log
│   ├── sweep_error_12345_0.log
│   └── ...
└── combinations/                 # Results (copied from scratch)
    ├── combo_hidden_size_10_set_seed_1/
    ├── combo_hidden_size_10_set_seed_2/
    ├── combo_hidden_size_20_set_seed_1/
    └── combo_hidden_size_20_set_seed_2/
```

## Example

Given this sweep configuration:

```yaml
# sweep.yaml
control:
  hidden_size: [10, 20, 30]
  set_seed: [1, 2, 3, 4, 5]
```

This creates 15 combinations (3 × 5). With `--max_concurrent_jobs 6`, up to 6 jobs run simultaneously.

## Monitoring Progress

### Using SLURM Commands

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Check job history
sacct -j <job_id>
```

### Checking Results

```bash
# Count completed combinations
ls experiments/training/my_experiment/combinations/ | wc -l

# Check for results files
find experiments/training/my_experiment/combinations/ -name "results.csv" | wc -l
```

## Migration Guide

### From Sequential to Parallel

**Before (Sequential):**
```python
@combination_sweep(exp_dir, mode="combination")
def run_sweep(config, save_dir):
    train_teacher_student(config=config, save_dir=save_dir, data_dir=data_dir, cluster=cluster)
```

**After (Parallel):**
```python
@parallel_combination_sweep(exp_dir, mode="combination", max_concurrent_jobs=6)
def run_sweep(config, save_dir):
    train_teacher_student(config=config, save_dir=save_dir, data_dir=data_dir, cluster=cluster)
```

Or simply use the CLI flag: `--parallel True`

## Best Practices

1. **Start Small**: Test with a small sweep first (e.g., 2×2 combinations)
2. **Resource Planning**: Consider your cluster's limits when setting `max_concurrent_jobs`
3. **Walltime**: Set appropriate walltime based on your experiment duration
4. **Monitoring**: Check SLURM logs regularly for any issues
5. **Results Management**: Results stay in scratch during execution - plan for data management

## Troubleshooting

### Common Issues

1. **Jobs not starting**: Check cluster queue status and resource availability
2. **Import errors**: Ensure the Python path is correctly set in the generated script
3. **Out of memory**: Adjust `--mem_per_cpu` or `--gpu_mem` parameters
4. **Timeout**: Increase `--walltime` for longer experiments

### Debug Mode

For testing, use `--submit_jobs False` to generate scripts without submitting:

```bash
python teacher_student/cli.py train --exp_id test --parallel True --submit_jobs False
```

This creates all files but doesn't submit to SLURM, allowing you to inspect the generated script.

## Compatibility

- **Existing Projects**: All existing code continues to work unchanged
- **Mixed Usage**: You can use parallel sweeps for some experiments and sequential for others
- **Cluster Requirements**: Requires SLURM job scheduler
- **Python Environment**: Uses the same virtual environment as sequential runs
