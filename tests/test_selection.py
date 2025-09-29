from utilities.ml_processes import select_best_model


def test_select_best_model_with_tiebreaker():
    # Define two candidates with close AUCs
    candidates = [
        {
            'model': 'xgb',
            'metrics': {'auc_roc': 0.8, 'recall': 0.6, 'precision': 0.75},
            'model_uri': 'models:/xgb/1',
        },
        {
            'model': 'rf',
            'metrics': {'auc_roc': 0.79, 'recall': 0.7, 'precision': 0.7},
            'model_uri': 'models:/rf/1',
        },
    ]
    selection_criteria = {
        'primary': 'auc_roc',
        'tiebreaker': [{'metric': 'recall', 'equality_threshold': 0.02}],
        'min_threshold': 0.5,
    }

    best = select_best_model(candidates, selection_criteria)
    # They are close in AUC (0.8 vs 0.79 difference 0.01 < 0.02), tie-breaker uses recall
    assert best['model'] == 'rf'