from azure.ai.ml import dsl, Input
from azure.ai.ml import load_component

# Load components
preprocess_component = load_component(source="../components/preprocess_dataset/preprocess_dataset.yaml")
train_component = load_component(source="../components/train/train.yaml")
evaluate_component = load_component(source="../components/evaluate/evaluate.yaml")


@dsl.pipeline(
    compute="cpu-cluster",  # AML compute target
    description="Credit Scoring Pipeline"
)
def credit_scoring_pipeline(raw_data_path: str):
    preprocess_step = preprocess_component(
        input_data=Input(type="uri_file", path=raw_data_path)
    )

    train_step = train_component(
        input_data=preprocess_step.outputs.train_output,
        config="./configs/training_candidates.yaml",
        num_cols="age,income",
        simple_cat_cols="gender,region",
        complex_cat_cols="occupation",
        passthrough_cols="account_length"
    )

    evaluate_step = evaluate_component()

    return {
        "train_output": preprocess_step.outputs.train_output,
        "test_output": preprocess_step.outputs.test_output,
        "metrics": evaluate_step.outputs.metrics_output,
        "best_model_uri": evaluate_step.outputs.best_model_uri,
    }
