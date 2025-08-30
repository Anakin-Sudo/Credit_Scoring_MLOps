from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import PipelineJob
from credit_scoring_pipeline import credit_scoring_pipeline


if __name__ == "__main__":
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="YOUR_SUBSCRIPTION_ID",
        resource_group_name="YOUR_RG",
        workspace_name="YOUR_WORKSPACE"
    )

    # Build pipeline job instance
    pipeline_job: PipelineJob = credit_scoring_pipeline(
        raw_data_path="azureml://datastores/workspaceblobstore/paths/raw_data/credit.csv"
    )

    # Submit to AML
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="credit_scoring_pipeline"
    )

    print(f"Pipeline job submitted: {pipeline_job.name}")
