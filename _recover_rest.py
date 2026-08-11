"""Resume cheater recovery: attention eval where dag_metrics is degenerate,
ATE eval where the CSV is missing. Guarded for Windows multiprocessing spawn."""
import json
import time

from _recover_cheater import _find_runs, _patch_config, _restore_config

SKIP = {"cheater_ds_scm1_continuous_dag_0_model_1"}  # fully evaluated already


def _needs_dag(run):
    dag = run / "eval/eval_attention_scores/files/dag_metrics.json"
    if not dag.exists():
        return True
    try:
        return not json.loads(dag.read_text()).get("standard_shd_cross")
    except Exception:
        return True


def main():
    from causaliT.evaluation.eval_funs.eval_attention import eval_attention_scores
    from causaliT.evaluation.eval_funs.eval_interventions import eval_interventions

    for run in _find_runs():
        if run.name in SKIP:
            continue
        need_ate = not (run / "eval/eval_ate_mc/files/ate_metrics_mc.csv").exists()
        need_dag = _needs_dag(run)
        if not (need_ate or need_dag):
            continue
        t0 = time.time()
        print(f"RUN {run.name} (ate={need_ate}, dag={need_dag})", flush=True)
        orig = _patch_config(run)
        try:
            if need_dag:
                eval_attention_scores(str(run), show_plots=False)
            if need_ate:
                eval_interventions(str(run), show_plots=False)
        except Exception as exc:
            print(f"  [FAIL] {run.name}: {exc}", flush=True)
        finally:
            _restore_config(run, orig)
        print(f"DONE {run.name} ({time.time() - t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
