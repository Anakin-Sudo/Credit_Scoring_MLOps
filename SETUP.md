# Setup Guide for Credit Scoring ML Pipeline

This guide provides detailed steps to configure your Azure environment, prepare the codebase and run the credit‑scoring pipeline.  Follow each section in sequence to ensure that resources are provisioned correctly and that secrets are handled securely.  The instructions assume you are working from the `Credit_Scoring/` directory and that your local time zone is Europe/Paris.

## 1 – Install Required Tools

Before beginning, install the following on your local machine:

* **Python 3.10+** – used for running the ML pipeline and scripts.
* **Azure CLI (v2)** – command‑line interface for managing Azure resources.  Follow Microsoft’s installation guide for your platform.
* **Azure ML extension for Azure CLI:** run `az extension add -n ml`.  This extension provides commands such as `az ml workspace`, `az ml environment`, and `az ml compute`.
* **Git** – to clone and manage the repository.
* **Conda (optional)** – for creating an isolated Python environment.  You can also use `pip` with `requirements.txt`.

Verify that the ML extension is installed by running `az ml -h`.  If not, ensure you have CLI version 2.15.0 or higher; the extension will be installed automatically when you first call an ML command.

## 2 – Clone the Repository and Set Up Python

1. Clone the project and change directory into it:

   ```sh
   git clone https://github.com/your-org/credit-scoring.git
   cd credit-scoring/Credit_Scoring
   ```

2. Create a Python environment.  Either use conda:

   ```sh
   conda env create -f environments/conda_dependencies.yml
   conda activate credit-scoring
   ```

   or install dependencies via pip:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. (Optional) Curate the dataset: Run the curation script to convert `german.data` into `german_credit.csv` with decoded categories and a binary `CreditRisk` target.  Only run this once, as the curated CSV is already included.

   ```sh
   python curate_german_data.py
   ```

## 3 – Populate the `.env` File

The repository includes `env.example` with non‑sensitive placeholders.  Copy it to `.env`:

```sh
cp env.example .env
```

Open `.env` in your editor and replace placeholders with real values.  Below is a description of each variable:

| Variable | Example value | Purpose |
|---|---|---|
| `SUBSCRIPTION_ID` | `00000000-1234-5678-90ab-cdef01234567` | Azure subscription where resources will be created |
| `TENANT_ID` | `11111111-2222-3333-4444-555555555555` | Azure AD tenant ID |
| `RESOURCE_GROUP` | `credit-scoring-rg` | Resource group for the ML workspace and compute |
| `WORKSPACE_NAME` | `credit-scoring-ws` | Name of the Azure ML workspace |
| `LOCATION` | `westeurope` | Azure region for all resources |
| `AZURE_STORAGE_CONTAINER_NAME` | `credit-scoring` | Optional: name of a blob container for data storage |
| `KEYVAULT_NAME` | `creditkv` | Optional: Key Vault to store secrets (leave blank if unused) |
| `KEYVAULT_URI` | `https://creditkv.vault.azure.net/` | URI of the Key Vault |
| `STORAGE_ACCOUNT_CONN_STR_SECRET_NAME` | `storage-conn-string` | Name of the secret in Key Vault containing the storage account key |
| `COMPUTE_NAME` | `cpu-cluster` | Name for the Azure ML compute cluster |
| `VM_SIZE` | `STANDARD_DS2_V2` | VM SKU for the compute cluster |
| `MIN_INSTANCES` | `0` | Minimum number of nodes in the cluster (saving costs when idle) |
| `MAX_INSTANCES` | `2` | Maximum number of nodes (scales out under load) |

These values are used by the provisioning scripts.  Do not include sensitive values (client secrets) here.

## 4 – Authenticate to Azure

Before running any `az` commands, authenticate to Azure.  The `bash_scripts/az_login.sh` and `set_az_defaults.sh` scripts call `az login` with the tenant ID read from `.env`.  You can run either script:

```sh
bash bash_scripts/az_login.sh
```

or run manually:

```sh
az login --tenant "$TENANT_ID"
az account set --subscription "$SUBSCRIPTION_ID"
```

You should see a browser window prompting you to sign in.  After authenticating, Azure CLI commands will run in the context of your subscription.

## 5 – Create the Resource Group and ML Workspace

Run the `create_az_ml_workspace.sh` script to create the resource group and Azure ML workspace defined in `.env`:

```sh
bash bash_scripts/create_az_ml_workspace.sh
```

This script uses `az group create` and `az ml workspace create`.  The ML workspace stores experiments, datasets, models and compute resources.

## 6 – Create or Update the Azure ML Environment

An Azure ML environment encapsulates the software environment used for training and inference.  The `create_az_env.sh` script uses `az ml environment create` to build or update an environment from `environments/env.yml`.  Run:

```sh
bash bash_scripts/create_az_env.sh
```

The YAML file defines the base Docker image, Python version and package dependencies.  When you run this script for the first time, it registers a new environment in your workspace; subsequent runs update the version if there are changes.

## 7 – Provision the Compute Cluster

Run the `create_compute.sh` script to create an Azure ML compute cluster for training.  The script reads `COMPUTE_NAME`, `VM_SIZE`, `MIN_INSTANCES` and `MAX_INSTANCES` from `.env` and calls `az ml compute create` with those parameters:

```sh
bash bash_scripts/create_compute.sh
```

The resulting cluster will scale from `MIN_INSTANCES` to `MAX_INSTANCES` nodes.  In the Azure ML CLI, a compute cluster definition includes fields such as `type: amlcompute`, `size`, `min_instances` and `max_instances`.  Keeping `MIN_INSTANCES` at zero ensures that you do not pay for idle nodes.

## 8 – Create a Service Principal for CI/CD

To allow automated workflows (e.g., GitHub Actions) to perform operations on your Azure resources, you need a service principal.  Use the `create_service_principal.sh` script:

```sh
export SP_NAME="credit-scoring-sp"
bash bash_scripts/create_service_principal.sh
```

This script calls `az ad sp create-for-rbac` with the scope set to your ML workspace and writes the credentials to `sp_output.json`.  **Do not commit this file to version control.**  Instead, open it and copy the values you need:

* **clientId** – Service principal application (client) ID
* **tenantId** – Directory (tenant) ID
* **subscriptionId** – Your subscription ID (already known)
* **clientSecret** – Secret used to authenticate

These identifiers should be stored as GitHub Secrets rather than in your `.env` file.  See section 11 for details.

## 9 – Assign Roles to the Service Principal

The service principal must have appropriate permissions to your ML workspace.  Run the `assign_service_principal_role.sh` script:

```sh
export SP_APP_ID="<clientId-from-sp_output.json>"
bash bash_scripts/assign_service_principal_role.sh
```

This script fetches the workspace resource ID and assigns the `AzureML Data Scientist` role to the principal.  You can adjust the role if you need more restrictive permissions.

If you plan to use Key Vault for secrets, run `assign_key_vault_role.sh` to grant the compute cluster’s managed identity access to secrets:

```sh
bash bash_scripts/assign_key_vault_role.sh
```

This script queries the compute’s principal ID and assigns the `Key Vault Secrets User` role on your Key Vault.

## 10 – Create the Compute Environment

Optional: If you plan to run jobs that require custom libraries (beyond those specified in `env.yml`), update `env.yml` and rerun `create_az_env.sh`.  Azure ML environments support versioning, so you can pin an environment to a specific version for reproducibility.  For details about `az ml environment create`, see Microsoft’s documentation.

## 11 – Store Credentials in GitHub Secrets

Do not place the service principal secret (`clientSecret`) in your repository.  Instead, store the following values in GitHub Secrets:

1. Navigate to your GitHub repository’s **Settings** > **Security** > **Secrets and variables** > **Actions** and click **New repository secret**.
2. Add secrets named:
   - `AZURE_CLIENT_ID` – the clientId from `sp_output.json`.
   - `AZURE_TENANT_ID` – your Azure AD tenant ID.
   - `AZURE_SUBSCRIPTION_ID` – your subscription ID.
   - `AZURE_CLIENT_SECRET` – the clientSecret from `sp_output.json` (if you choose to use secret‑based authentication instead of OIDC).  Omit this if you configure a federated identity.
3. For public repositories, consider using **environment secrets** for additional approval controls.

To avoid using secrets entirely, you can configure a **federated identity credential** on your service principal or use a **user‑assigned managed identity**.  In this case, GitHub Actions uses OpenID Connect to obtain a token.  See the Azure docs for prerequisites.
## 12 – Use the Provided GitHub Actions Workflows

This repository already contains two workflows under `.github/workflows/`:

* **`ci.yml`** – runs unit tests when code is pushed to or opened as a pull request against the `main` branch.
* **`azure-ml-pipeline.yml`** – submits the credit scoring pipeline to Azure ML.  The workflow reads your Azure identifiers from secrets, writes them to `.env`, then invokes the `pipelines/submit_pipeline.py` script.

To enable these workflows:

1. **Add repository secrets:** Using your browser, open the **Settings** page of your GitHub repository and navigate to **Security → Secrets and variables → Actions**.  Create the secrets listed in section 11 (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME`, `AZURE_COMPUTE_NAME`).  If you use secret‑based authentication instead of OIDC, also provide `AZURE_CLIENT_SECRET`.  The Azure login action uses these values to authenticate to Azure.

2. **Verify the workflow inputs:** The `azure-ml-pipeline.yml` workflow accepts a `data_blob` input when triggered manually.  By default it writes the blob name to `.env` as `BLOB_NAME`.  You can modify the default input or hard‑code your blob name within the workflow.

3. **Trigger the workflow:** From the **Actions** tab in GitHub, select **AzureML Pipeline**, click **Run workflow**, and enter the name of your data blob.  The workflow will log into Azure, install dependencies and submit the pipeline job.  Because the workflow uses managed identities and OIDC, you do not need to embed secrets in files.

There is no need to create an additional workflow file; the provided `ci.yml` and `azure-ml-pipeline.yml` are sufficient for most scenarios.  If your organisation requires additional steps (for example, model registration or deployment), you can extend these workflows using the same authentication pattern.

## 13 – Submit the Pipeline Locally (Alternative)

If you prefer to run the pipeline from your local machine instead of GitHub Actions, simply activate your Python environment, ensure you are logged into Azure via `az login`, and run:

```sh
python pipelines/submit_pipeline.py
```

This script uses `DefaultAzureCredential` from the Azure SDK to authenticate.  On your local machine it will fall back to the CLI credential if you are logged in.

## 14 – Cleanup Resources (Optional)

To delete the resources created by this project (workspace, compute, environment), you can run the following commands:

```sh
az ml compute delete --name "$COMPUTE_NAME" --workspace-name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP"
az ml workspace delete --name "$WORKSPACE_NAME" --resource-group "$RESOURCE_GROUP" --yes
az group delete --name "$RESOURCE_GROUP" --yes --no-wait
```

Be careful with these commands; they permanently remove resources and may incur charges if left running.

## 15 – Troubleshooting

- **Login failures:** Ensure your tenant ID and subscription ID are correct and that you have sufficient permissions.  The Azure Login action in GitHub requires the `id-token` permission for OIDC.
- **Compute quota errors:** Some VM sizes may not be available in your region or subscription.  Adjust `VM_SIZE` in `.env` or request a quota increase via the Azure portal.
- **Invalid secret:** If `AZURE_CLIENT_SECRET` expires or is incorrect, refresh the secret in `sp_output.json` and update the GitHub secret.
- **Pipeline submission errors:** Verify that your YAML config files under `configs/` are valid and that the dataset path (`RAW_DATA_PATH` in `.env`) points to a file accessible to Azure ML (e.g., an Azure Storage blob or public URL).  You can upload your curated CSV to a storage container and reference its URI.

By following the above steps, you will have a fully provisioned Azure Machine Learning environment and a secure CI/CD setup for building and deploying credit‑scoring models.  Consult the main README for an overview and the official Azure documentation for additional guidance on compute clusters and secure authentication from GitHub.
