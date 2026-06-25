"""Main extraction orchestrator for MLflow workspace model registry metadata.

Targets the WORKSPACE registry (not Unity Catalog).
Enumerates all registered models and extracts metadata in parallel.
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


def extract_all_metadata(max_workers=10, include_run_params=True,
                         include_run_metrics=True, verbose=True):
    # Use workspace registry (NOT UC)
    client = MlflowClient(registry_uri="databricks")
    ctx = get_workspace_context()
    workspace_host = ctx["workspace_host"]

    if verbose:
        print(f"Workspace: {workspace_host}")
        print(f"Registry: workspace (legacy MLflow)")

    models = enumerate_all_models(client)
    if verbose:
        print(f"Total registered models found: {len(models)}")

    all_rows, errors = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_model, client, m, workspace_host, None,
                None, include_run_params, include_run_metrics
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
