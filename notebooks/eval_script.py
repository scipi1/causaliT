from os.path import dirname, abspath, join
import sys
import pandas as pd

root_path = dirname(abspath("./"))
import sys
sys.path.append(root_path)
from causaliT.evaluation.predict import *
from causaliT.evaluation.eval_sweeps import *
from causaliT.paths import ROOT_DIR



def evaluate_do_interventions(exp_path, filepath):

    df_base = predict_nested_all_checkpoints(exp_path, input_conditioning_fn=None)
    df_base["intervention"] = "baseline"

    intervention_fn_S1 = create_intervention_fn(interventions={1: 0.0})
    df_S1 = predict_nested_all_checkpoints(exp_path, input_conditioning_fn=intervention_fn_S1)
    df_S1["intervention"] = "S1=0"

    intervention_fn_S2 = create_intervention_fn(interventions={2: 0.0})
    df_S2 = predict_nested_all_checkpoints(exp_path, input_conditioning_fn=intervention_fn_S2)
    df_S2["intervention"] = "S2=0"

    intervention_fn_S3 = create_intervention_fn(interventions={3: 0.0})
    df_S3 = predict_nested_all_checkpoints(exp_path, input_conditioning_fn=intervention_fn_S3)
    df_S3["intervention"] = "S3=0"

    df = pd.concat([df_base, df_S1, df_S2, df_S3])

    df['epoch'] = df['checkpoint_name'].str.extract(r'epoch=(\d+)').astype(int)
    df['model'] = df['model_folder'].str.extract(r'stage_(Lie|SoftMax)')
    df = df.drop(columns=["checkpoint_name", "model_folder", "level_0"])
    df.to_csv(filepath)
    


if __name__ == "__main__":
    filepath = "eval/intervention_scm6.csv"
    exp_path = join(ROOT_DIR,"experiments/euler_scm6")
    evaluate_do_interventions(exp_path, filepath)
