from azure.ai.ml import dsl, Input
from azure.ai.ml import load_component

# Load components
preprocess_component = load_component(source="../components/preprocess_dataset/preprocess_dataset.yaml")
train_component = load_component(source="../components/train/train.yaml")
evaluate_component = load_component(source="../components/evaluate/evaluate.yaml")  # placeholder


@dsl.pipeline(
    description="Credit Scoring Pipeline"
)
def credit_scoring_pipeline(
    raw_data_path: str,
    candidates_json: str,
    feature_groups_json: str,
    selection_criteria_json: str,
    dropna_cols: str = None,
    drop_duplicates: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "CreditRisk",
    cv_folds: int = 5,
    decision_threshold: float = 0.05,
):
    # Step 1: Preprocessing
    preprocess_step = preprocess_component(
        input_data=Input(type="uri_file", path=raw_data_path),
        dropna_cols=dropna_cols,
        drop_duplicates=drop_duplicates,
        test_size=test_size,
        random_state=random_state,
        stratify_col=stratify_col,
    )

    # Step 2: Training
    train_step = train_component(
        input_data=preprocess_step.outputs.train_output,
        candidates=candidates_json,
        feature_groups=feature_groups_json,
        random_state=random_state,
        cv_folds=cv_folds,
        threshold=decision_threshold,
    )

    # Step 3: Evaluation
    evaluate_step = evaluate_component(
        test_data=preprocess_step.outputs.test_output,
        selection_criteria=selection_criteria_json,
        threshold=decision_threshold,
    )

    return {
        "train_data": preprocess_step.outputs.train_output,
        "test_data": preprocess_step.outputs.test_output,
        "metrics": evaluate_step.outputs.metrics_output,
        "best_model_pointer_file": evaluate_step.outputs.best_model_pointer_file,
    }
