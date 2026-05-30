import os
from kfp.client import Client

def trigger_pipeline():
    n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
    test_size = float(os.environ.get("TEST_SIZE", 0.2))
    model_type = os.environ.get("MODEL_TYPE", "RandomForestClassifier")

    client = Client()

    pipeline_arguments = {
        "n_estimators": n_estimators,
        "test_size": test_size,
        "model_type": model_type
    }

    run_result = client.create_run_from_pipeline_package(
        pipeline_file='iris_pipeline.yaml',
        arguments=pipeline_arguments,
        experiment_name='Auto-GitOps-Experiment',
        enable_caching=False
    )

if __name__ == "__main__":
    trigger_pipeline()