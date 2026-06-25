"""Main extraction orchestrator for MLflow model registry metadata.

Enumerates all registered models, extracts metadata in parallel,
and maps models to serving endpoints for inference table linkage.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from mlflow_utils import process_model


def get_workspace_context():
    w = WorkspaceClient()
    return {"workspace_host": w.config.host, "w_client": w}


def enumerate_all_models(client: MlflowClient):
    models, token = [], None
    while True:
        page = client.search_registered_models(max_results=100, page_token=token)
        models.extend(page)
        token = page.token if hasattr(page, "token") else None
        if not token:
            break
    return models


def get_endpoint_model_map(w_client) -> dict:
    """Map model full names to their serving endpoint names."""
    mapping = {}
    try:
        for ep in w_client.serving_endpoints.list():
            try:
                detail = w_client.serving_endpoints.get(ep.name)
                if detail.config:
                    entities = getattr(detail.config, "served_entities", None) or []
                    for se in entities:
                        name = getattr(se, "entity_name", None)
                        if name:
                            mapping[name] = {"endpoint_name": ep.name,
                                             "creator": getattr(detail, "creator", None)}
            except Exception:
                pass
    except Exception:
        pass
    return mapping


def extract_all_metadata(max_workers=10, include_run_params=True,
                         include_run_metrics=True, verbose=True):
    client = MlflowClient()
    ctx = get_workspace_context()
    workspace_host = ctx["workspace_host"]
    w_client = ctx["w_client"]

    if verbose:
        print(f"Workspace: {workspace_host}")

    models = enumerate_all_models(client)
    if verbose:
        print(f"Total registered models found: {len(models)}")

    # Build endpoint mapping
    endpoint_map = get_endpoint_model_map(w_client)

    all_rows, errors = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_model, client, m, workspace_host, w_client,
                endpoint_map, include_run_params, include_run_metrics
            ): m.name for m in models
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows = future.result()
                all_rows.extend(rows)
                errors += sum(1 for r in rows if r.get("extraction_error"))
            except Exception:
                errors += 1
            if verbose and i % 50 == 0:
                print(f"  Processed {i}/{len(models)} models ({len(all_rows)} rows)...")

    if verbose:
        print(f"\nDone! Models: {len(models)} | Rows: {len(all_rows)} | Errors: {errors}")

    stats = {"total_models": len(models), "total_rows": len(all_rows), "errors": errors}
    return all_rows, ctx, stats
