# Credit Scoring ML Pipeline on Azure

This repository implements an end‑to‑end **credit‑scoring** solution built on top of [Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/).  The project demonstrates how to curate a raw dataset, preprocess it, train multiple candidate models, evaluate and select a champion model, and operationalise the workflow using Azure ML pipelines.  Infrastructure provisioning (resource group, workspace, compute cluster and managed identities) is automated via Bash scripts.  Out‑of‑the‑box GitHub Actions workflows live under `.github/workflows/` to run tests and submit the pipeline; there is no need to hand‑roll a workflow yourself – simply configure secrets and trigger the existing ones.

The pipeline uses the classic German Credit dataset, curated into a CSV file and enriched with decoded categorical variables.  The ML workflow is modular, with each stage defined as a separate component and orchestrated by a DSL pipeline.  The repository is organised to encourage reproducibility, security and clear separation between infrastructure code and ML logic.

## Key Features

- **Data curation script** (`curate_german_data.py`): converts the raw UCI German Credit `.data` file into a clean CSV with descriptive column names and binary target labels.
- **Modular components**: ingest, preprocess, train and evaluate components are defined under `components/` with corresponding YAML specifications.  These can be reused or swapped out for different datasets or models.
- **DSL pipeline** (`pipelines/pipelines.py`): composes the components into a single pipeline with inputs for raw data, model candidates and selection criteria.
- **Pipeline submission script** (`pipelines/submit_pipeline.py`): reads configuration files (`configs/preprocess_config.yaml`, `configs/feature_groups.yaml`, `configs/training_config.yaml`), serialises them to JSON and submits the pipeline to Azure ML.
- **Infrastructure automation** (`bash_scripts/`): Bash scripts wrap Azure CLI/ML CLI commands to log in, create or update the ML workspace, environment, compute cluster and service principal, and assign roles.
- **Environment definitions** (`environments/env.yml` and `environments/conda_dependencies.yml`): define the runtime environment for your ML jobs and the conda dependencies for local development.
- **Example configuration files**: YAML files under `configs/` specify preprocessing parameters, feature groupings and candidate models (e.g., logistic regression, random forest, XGBoost) with hyperparameters.

## Repository Structure

```
Credit_Scoring/
│  german.data                    # Raw UCI dataset (numeric and coded categorical values)
│  german_credit.csv              # Curated CSV (generated from raw)
│  env.example                    # Template for environment variables (non‑sensitive)
│  requirements.txt               # Python dependencies for local development
│
├─ bash_scripts/                  # Bash helpers for Azure provisioning
│   ├─ az_login.sh                # Interactive login to Azure using tenant ID
│   ├─ set_az_defaults.sh         # Log in with tenant context (az login) and set defaults
│   ├─ create_az_ml_workspace.sh  # Create resource group and Azure ML workspace
│   ├─ create_az_env.sh           # Create or update an Azure ML environment
│   ├─ create_compute.sh          # Provision an Azure ML compute cluster
│   ├─ create_service_principal.sh# Create a service principal for CI/CD
│   ├─ assign_service_principal_role.sh # Grant the service principal access to the workspace
│   └─ assign_key_vault_role.sh   # Assign Key Vault secrets access to the compute identity
│
├─ components/                   # Reusable ML components (code + YAML)
│   ├─ ingest/
│   ├─ preprocess_dataset/
│   ├─ train/
│   └─ evaluate/
│
├─ configs/                      # Configuration files for preprocessing and training
│   ├─ preprocess_config.yaml
│   ├─ feature_groups.yaml
│   └─ training_config.yaml
│
├─ environments/                 # Azure ML environment definitions
│   ├─ env.yml                   # YAML specifying docker base image and pip dependencies
│   └─ conda_dependencies.yml    # Conda environment for local development
│
├─ pipelines/                    # Pipeline definitions and submission script
│   ├─ pipelines.py              # DSL pipeline combining preprocessing, training and evaluation
│   └─ submit_pipeline.py        # Script to serialise configs and submit the pipeline
│
├─ notebooks/                    # (Optional) notebooks for exploration and demonstration
│
├─ utilities/                    # Helper modules (data prep, ML processes, MLflow integration)
│   ├─ ml_processes.py
│   ├─ mlflow_processes.py
│   ├─ model_factory.py
│   └─ azure_storage.py
│
└─ tests/                        # Unit tests for utility functions and components

```

## Quick Start

1. **Install prerequisites:** Install [Python 3.10+](https://www.python.org/), the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) with the ML extension (`az extension add -n ml`), and optionally [conda](https://docs.conda.io/en/latest/) to create an isolated environment.  Ensure you have rights to create resources in an Azure subscription.

2. **Clone the repository:**

   ```sh
   git clone https://github.com/your-org/credit-scoring.git
   cd credit-scoring/Credit_Scoring
   ```

3. **Prepare your Python environment:** Create a conda environment from `environments/conda_dependencies.yml` or install dependencies with `pip install -r requirements.txt`.  This will install all libraries required by the components and pipeline.

4. **Curate the dataset (optional):** The repository includes a curated `german_credit.csv`.  If you wish to regenerate it from the raw UCI file (`german.data`), run:

   ```sh
   python curate_german_data.py
   ```

5. **Consult the setup guide:** For instructions on populating `.env`, running the provisioning scripts, creating a service principal and managing secrets, refer to [SETUP.md](SETUP.md).  The setup guide explains how to create the Azure ML workspace, environment and compute cluster, and how to prepare the Azure secrets used by the provided workflows.

6. **Use the built‑in workflows:** This project already defines two workflows under `.github/workflows/`:
   * `ci.yml` runs unit tests when code is pushed to or merged into the `main` branch.
   * `azure‑ml‑pipeline.yml` submits the pipeline job to Azure ML.  It accepts a `data_blob` input when triggered manually and reads your Azure identifiers from repository secrets.  To use it, simply configure secrets such as `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME` and `AZURE_COMPUTE_NAME` in the repository settings as described in the setup guide.  You can then trigger the workflow via the GitHub Actions tab.

## Why Use Managed Identities and GitHub Secrets?

Running machine learning workloads in the cloud requires authenticating to Azure resources.  Storing secrets in files or environment variables is risky.  Microsoft recommends creating a **service principal** or **user‑assigned managed identity**, granting it access to your workspace and storing its identifiers in GitHub Secrets.  In your workflow, the Azure Login action requests an ID token to authenticate with Azure. This avoids the need to embed secrets in your code or `.env` files.

## Next Steps

Read the [Setup guide](SETUP.md) for detailed instructions on provisioning Azure resources, creating a service principal or managed identity, assigning roles, populating the `.env` file, and configuring GitHub Actions.  See also the Azure documentation on creating compute clusters and authenticating from GitHub to Azure.
