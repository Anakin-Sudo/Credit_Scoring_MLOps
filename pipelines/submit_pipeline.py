import os
import json
import yaml
from dotenv import load_dotenv
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

from pipelines import credit_scoring_pipeline

# --- 1. Load environment variables ---
load_dotenv()

SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
COMPUTE_NAME = os.getenv("COMPUTE_NAME")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")

# --- 2. Connect to Azure ML workspace ---
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# --- 3. Load training config ---
with open("./configs/training_config.yaml", "r") as f:
    training_config = yaml.safe_load(f)

candidates = training_config["candidates"]
global_params = training_config["global_hyperparams"]
selection_criteria = training_config["selection_criteria"]

# Serialize configs to JSON strings
candidates_json = json.dumps(candidates)
selection_criteria_json = json.dumps(selection_criteria)

# --- 4. Load feature groups config ---
with open("./configs/feature_groups.yaml", "r") as f:
    feature_groups = yaml.safe_load(f)

feature_groups_json = json.dumps(feature_groups)

# --- 5. Load preprocess config ---
with open("./configs/preprocess_config.yaml", "r") as f:
    preprocess_config = yaml.safe_load(f)

split_cfg = preprocess_config["split"]
clean_cfg = preprocess_config["cleaning"]

# --- 6. Build pipeline job ---
pipeline_job = credit_scoring_pipeline(
    raw_data_path=RAW_DATA_PATH,
    candidates_json=candidates_json,
    feature_groups_json=feature_groups_json,
    selection_criteria_json=selection_criteria_json,  # NEW
    dropna_cols=",".join(clean_cfg["dropna_cols"]) if clean_cfg["dropna_cols"] else None,
    drop_duplicates=clean_cfg["drop_duplicates"],
    test_size=split_cfg["test_size"],
    random_state=split_cfg["random_state"],
    stratify_col=split_cfg["stratify_col"],
    cv_folds=global_params.get("cv_folds"),
    decision_threshold=global_params.get("decision_threshold"),
)

# Attach compute target explicitly
pipeline_job.settings.default_compute = COMPUTE_NAME

# --- 7. Submit the pipeline job ---
submitted_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="credit_scoring_experiment",
)

print(f"Pipeline submitted. Job name: {submitted_job.name}")
