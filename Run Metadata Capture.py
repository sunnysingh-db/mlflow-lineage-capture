# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Run Extraction, Write Delta Table, Generate CSV
# --- Suppress all warnings, progress bars, and widget noise ---
import warnings, logging, os, sys
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TQDM_DISABLE"] = "1"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
for name in ["pyspark.sql.connect", "mlflow", "py4j", "urllib3", "databricks.sdk"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)
try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except Exception:
    pass

# --- Discover current user & paths (zero hardcoding) ---
import importlib
from datetime import datetime
import pandas as pd
from pyspark.sql.functions import current_timestamp

current_user = spark.sql("SELECT current_user()").first()[0]
user_safe = current_user.split("@")[0].replace(".", "_").replace("-", "_")
DATABASE_NAME = f"users.{user_safe}"

# Discover project directory: find the folder containing BOTH co-located modules.
# Handles stale sys.path entries from other sessions by prioritizing the correct dir.
for _m in ["mlflow_utils", "extraction"]:
    sys.modules.pop(_m, None)
_candidates = [p for p in sys.path
               if p and os.path.isfile(os.path.join(p, "mlflow_utils.py"))
               and os.path.isfile(os.path.join(p, "extraction.py"))]
# Remove ALL candidate paths, then re-insert the correct one at position 0
for _c in _candidates:
    while _c in sys.path:
        sys.path.remove(_c)
# The correct dir is the one Databricks added for THIS notebook (last added = last in list)
# In a fresh session there's only one; in stale sessions, the last is the auto-added one.
project_dir = _candidates[-1] if _candidates else os.getcwd()
sys.path.insert(0, project_dir)
os.chdir(project_dir)  # set CWD so relative paths work

import mlflow_utils, extraction
from extraction import extract_all_metadata

# --- Extract ---
rows, ctx, stats = extract_all_metadata(max_workers=10, verbose=True)

# --- Build DataFrame ---
df = spark.createDataFrame(pd.DataFrame(rows)).withColumn("extracted_at", current_timestamp())
df = df.select(*[c for c in df.columns if c != "extracted_at"] + ["extracted_at"])

# --- Write Delta ---
output_table = f"{DATABASE_NAME}.mlflow_model_registry_metadata"
try:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
except Exception:
    pass
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)

# --- Export CSV (relative path, written to notebook's own directory) ---
csv_filename = f"mlflow_registry_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.toPandas().to_csv(csv_filename, index=False)

# --- Summary ---
row_count = df.count()
print(f"\n{'='*60}")
print(f"  ✓ Delta: {output_table} — {row_count} rows, {len(df.columns)} cols")
print(f"  ✓ CSV: ./{csv_filename}")
print(f"  ✓ User: {current_user} | Models: {stats['total_models']} | Errors: {stats['errors']}")
print(f"{'='*60}")

# /Workspace/Users/... -> /files/Users/... for browser download
_download_url = os.path.join(project_dir, csv_filename).replace("/Workspace/", "/files/")
displayHTML(f'<a href="{_download_url}" target="_blank">📥 Download CSV</a>')
display(df)
