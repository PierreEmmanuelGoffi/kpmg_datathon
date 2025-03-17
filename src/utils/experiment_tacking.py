"""
Utilities for experiment tracking with MLflow.
"""

import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def setup_mlflow(experiment_name, tracking_uri=None):
    """
    Set up MLflow for experiment tracking.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: URI for MLflow tracking server (optional)

    Returns:
        experiment_id: ID of the experiment
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=os.path.join("mlruns", experiment_name),
        )
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def start_run(experiment_id, run_name=None, config=None):
    """
    Start a new MLflow run.

    Args:
        experiment_id: ID of the experiment
        run_name: Name for the run (optional, defaults to timestamp)
        config: Configuration dictionary to log as parameters (optional)

    Returns:
        run_id: ID of the started run
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Log config parameters
        if config is not None:
            # Flatten nested dictionaries for logging
            flat_params = _flatten_dict(config)
            for key, value in flat_params.items():
                mlflow.log_param(key, value)

        return run.info.run_id


def _flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary for MLflow parameter logging.

    Args:
        d: Dictionary to flatten
        parent_key: Key of parent dictionary (used for recursion)
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_model_performance(metrics, artifacts=None, model=None, model_name=None):
    """
    Log model performance metrics and artifacts.

    Args:
        metrics: Dictionary of metrics to log
        artifacts: Dictionary of file paths to log as artifacts (optional)
        model: Model object to log (optional)
        model_name: Name for the logged model (optional)
    """
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Log artifacts
    if artifacts is not None:
        for artifact_name, artifact_path in artifacts.items():
            mlflow.log_artifact(artifact_path, artifact_name)

    # Log model
    if model is not None and model_name is not None:
        mlflow.sklearn.log_model(model, model_name)


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_best_run(experiment_id, metric_name, mode="max"):
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_id: ID of the experiment
        metric_name: Name of the metric to optimize
        mode: 'max' or 'min' (whether higher or lower values are better)

    Returns:
        Best run information
    """
    client = MlflowClient()

    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"metrics.{metric_name} IS NOT NULL",
        order_by=[f"metrics.{metric_name} {'DESC' if mode == 'max' else 'ASC'}"],
    )

    if runs:
        return runs[0]
    else:
        return None
