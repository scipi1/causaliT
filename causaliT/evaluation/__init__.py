"""
Evaluation and prediction functions.
"""

from .predict import (
    predict,
    mk_quick_pred_plot,
    predict_test_from_ckpt,
    predict_test_from_ckpt_adaptive,
    create_predictor,
    create_input_blanking_fn,
)

from .predictors import (
    BasePredictor,
    PredictionResult,
    TransformerPredictor,
)

from .metrics import (
    compute_prediction_metrics,
    compare_predictions,
    aggregate_metrics,
)

__all__ = [
    # Main prediction functions
    'predict',
    'mk_quick_pred_plot',
    'predict_test_from_ckpt',
    'predict_test_from_ckpt_adaptive',
    'create_predictor',
    'create_input_blanking_fn',
    
    # Predictor classes
    'BasePredictor',
    'PredictionResult',
    'TransformerPredictor',
    
    # Metrics functions
    'compute_prediction_metrics',
    'compare_predictions',
    'aggregate_metrics',
]
