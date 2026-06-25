"""MLflow model metadata + lineage + user extraction utilities.

Targets the MLflow WORKSPACE model registry (not UC).
Extracts per-version: model identity, version details, run context,
lineage (signatures, feature tables, datasets), user metadata,
and training source.
"""

import json
import yaml
from mlflow.tracking import MlflowClient


def safe_get_tags(obj) -> dict:
    tags = getattr(obj, "tags", None)
    if tags is None:
        return {}
    if isinstance(tags, dict):
        return tags
    if isinstance(tags, list):
        return {t.key: t.value for t in tags}
    return {}


def safe_get_aliases(obj) -> list:
    aliases = getattr(obj, "aliases", None)
    if aliases is None:
        return []
    if isinstance(aliases, list):
        return aliases
    return []


def extract_lineage(client: MlflowClient, run_id: str, run=None) -> dict:
    """Extract lineage: signature, flavors, feature tables, datasets."""
    result = {
        "model_signature_inputs": None, "model_signature_outputs": None,
        "model_flavors": None, "feature_tables_json": None,
        "dataset_inputs_json": None, "datasets_tag_json": None,
        "loader_module": None,
    }
    if not run_id:
        return result

    # Flavors & loader from log-model.history tag
    if run:
        tags = run.data.tags
        history_raw = tags.get("mlflow.log-model.history")
        if history_raw:
            try:
                history = json.loads(history_raw)
                if history:
                    latest = history[-1]
                    flavors = latest.get("flavors", {})
                    result["model_flavors"] = json.dumps(list(flavors.keys()))
                    pyfunc = flavors.get("python_function", {})
                    result["loader_module"] = pyfunc.get("loader_module")
            except Exception:
                pass
        # mlflow.datasets tag
        ds_tag = tags.get("mlflow.datasets")
        if ds_tag:
            result["datasets_tag_json"] = ds_tag

    # Signature from MLmodel artifact
    try:
        ml_path = client.download_artifacts(run_id, "model/MLmodel")
        with open(ml_path) as f:
            mlmodel = yaml.safe_load(f)
        sig = mlmodel.get("signature")
        if sig:
            result["model_signature_inputs"] = sig.get("inputs")
            result["model_signature_outputs"] = sig.get("outputs")
        if not result["model_flavors"]:
            flavors = mlmodel.get("flavors", {})
            if flavors:
                result["model_flavors"] = json.dumps(list(flavors.keys()))
        if not result["loader_module"]:
            pyfunc = mlmodel.get("flavors", {}).get("python_function", {})
            result["loader_module"] = pyfunc.get("loader_module")
    except Exception:
        pass

    # Dataset inputs from MLflow Datasets API
    if run and hasattr(run, "inputs") and run.inputs:
        try:
            di_list = run.inputs.dataset_inputs if hasattr(run.inputs, "dataset_inputs") else []
            if di_list:
                result["dataset_inputs_json"] = json.dumps([
                    {"name": di.dataset.name, "digest": di.dataset.digest,
                     "source_type": di.dataset.source_type,
                     "source": di.dataset.source, "schema": di.dataset.schema}
                    for di in di_list
                ])
        except Exception:
            pass

    # Feature Store feature_spec.yaml
    if result["loader_module"] and "feature_store" in (result["loader_module"] or ""):
        try:
            local_path = client.download_artifacts(run_id, "model/data/feature_store/feature_spec.yaml")
            with open(local_path) as f:
                fs = yaml.safe_load(f)
            tables = []
            for entry in fs.get("input_tables", []):
                if isinstance(entry, dict):
                    for tname in entry:
                        tables.append({"table_name": tname, "features": [], "lookup_keys": []})
            for entry in fs.get("input_columns", []):
                if isinstance(entry, dict):
                    for col_name, info in entry.items():
                        if isinstance(info, dict) and info.get("source") == "feature_store":
                            tname = info.get("table_name", "")
                            for ft in tables:
                                if ft["table_name"] == tname:
                                    ft["features"].append(info.get("feature_name", col_name))
                                    for lk in info.get("lookup_key", []):
                                        if lk not in ft["lookup_keys"]:
                                            ft["lookup_keys"].append(lk)
                                    break
            if tables:
                result["feature_tables_json"] = json.dumps(tables)
        except Exception:
            pass

    return result


def get_model_version_metadata(
    client: MlflowClient, model_name: str, version_obj,
    workspace_host: str, model_description: str = "",
    include_run_params=True, include_run_metrics=True,
) -> dict:
    """Extract full metadata for a single model version (workspace registry)."""
    row = {
        "model_name": model_name,
        "model_description": model_description,
        "version_number": int(version_obj.version),
        "version_status": version_obj.status,
        "version_description": version_obj.description or "",
        "version_source": version_obj.source or "",
        "version_run_id": version_obj.run_id or "",
        "version_run_link": version_obj.run_link or "",
        "version_creation_ts": version_obj.creation_timestamp,
        "version_last_updated_ts": version_obj.last_updated_timestamp,
        "version_current_stage": getattr(version_obj, "current_stage", None),
        "version_tags": json.dumps(safe_get_tags(version_obj)),
        # User metadata
        "mlflow_user": None, "mlflow_notebook_path": None,
        "experiment_owner_email": None,
        "job_id": None, "job_run_id": None, "cluster_id": None,
        # Training source
        "automl_training_table": None, "automl_target_col": None,
        # Lineage
        "model_signature_inputs": None, "model_signature_outputs": None,
        "model_flavors": None, "feature_tables_json": None,
        "dataset_inputs_json": None, "datasets_tag_json": None,
        "loader_module": None,
        # Run details
        "run_experiment_id": None, "run_experiment_name": None,
        "run_artifact_uri": None, "run_name": None,
        "run_status": None, "run_user_id": None,
        "run_start_time": None, "run_end_time": None,
        "run_lifecycle_stage": None,
        "run_source_notebook_path": None, "run_source_type": None,
        "run_git_commit": None, "run_git_branch": None, "run_git_repo_url": None,
        "run_params_json": None, "run_metrics_json": None, "run_tags_json": None,
        "workspace_host": workspace_host,
        "extraction_error": None,
    }

    # Run details
    run = None
    if version_obj.run_id:
        try:
            run = client.get_run(version_obj.run_id)
            row["run_experiment_id"] = run.info.experiment_id
            row["run_artifact_uri"] = run.info.artifact_uri
            row["run_name"] = run.info.run_name
            row["run_status"] = run.info.status
            row["run_user_id"] = run.info.user_id
            row["run_start_time"] = run.info.start_time
            row["run_end_time"] = run.info.end_time
            row["run_lifecycle_stage"] = run.info.lifecycle_stage

            tags = run.data.tags
            row["run_source_notebook_path"] = tags.get(
                "mlflow.source.name", tags.get("mlflow.databricks.notebookPath", ""))
            row["run_source_type"] = tags.get("mlflow.source.type", "")
            row["run_git_commit"] = tags.get("mlflow.source.git.commit", "")
            row["run_git_branch"] = tags.get("mlflow.source.git.branch", "")
            row["run_git_repo_url"] = tags.get("mlflow.source.git.repoURL", "")
            # User metadata from run
            row["mlflow_user"] = tags.get("mlflow.user")
            row["mlflow_notebook_path"] = tags.get("mlflow.databricks.notebookPath")
            row["job_id"] = tags.get("mlflow.databricks.jobID")
            row["job_run_id"] = tags.get("mlflow.databricks.jobRunID")
            row["cluster_id"] = tags.get("mlflow.databricks.cluster.id")

            if include_run_params:
                row["run_params_json"] = json.dumps(run.data.params)
            if include_run_metrics:
                row["run_metrics_json"] = json.dumps(run.data.metrics)
            row["run_tags_json"] = json.dumps(tags)

            # Experiment details + owner
            try:
                exp = client.get_experiment(run.info.experiment_id)
                row["run_experiment_name"] = exp.name
                exp_tags = exp.tags or {}
                row["experiment_owner_email"] = exp_tags.get("mlflow.ownerEmail")
                row["automl_training_table"] = exp_tags.get("_databricks_automl.table_name")
                row["automl_target_col"] = exp_tags.get("_databricks_automl.target_col")
            except Exception:
                pass

        except Exception as e:
            row["extraction_error"] = str(e)[:200]

    # Lineage
    try:
        lineage = extract_lineage(client, version_obj.run_id, run)
        row.update(lineage)
    except Exception:
        pass

    return row


def process_model(
    client: MlflowClient, model, workspace_host: str,
    w_client=None, endpoint_map=None,
    include_run_params=True, include_run_metrics=True,
) -> list:
    """Process a registered model and all its versions (workspace registry)."""
    model_name = model.name
    model_desc = getattr(model, "description", "") or ""
    rows = []
    try:
        versions = client.search_model_versions(filter_string=f"name=\'{model_name}\'")
        if not versions:
            rows.append({"model_name": model_name, "model_description": model_desc,
                         "version_number": 0, "version_status": "NO_VERSIONS",
                         "workspace_host": workspace_host, "extraction_error": None})
        else:
            for v in versions:
                row = get_model_version_metadata(
                    client, model_name, v, workspace_host, model_desc,
                    include_run_params, include_run_metrics)
                rows.append(row)
    except Exception as e:
        rows.append({"model_name": model_name, "model_description": model_desc,
                     "version_number": -1, "version_status": "ERROR",
                     "extraction_error": str(e)[:200], "workspace_host": workspace_host})
    return rows
