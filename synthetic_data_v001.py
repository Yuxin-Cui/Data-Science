#!/usr/bin/env python3
# Generates synthetic data using SDV GaussianCopula.
# Default: do not save model or metadata unless flags are provided.

import argparse
import os
import re
import sys
import time
from datetime import datetime
import warnings

# Suppress SDV warnings BEFORE importing SDV
# 1) “strongly recommend saving the metadata using 'save_to_json'...”
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*strongly recommend saving the metadata using 'save_to_json'.*"
)
# 2) Previous PK overwrite notice
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*existing primary key.*will be removed.*"
)
# 3) SingleTableMetadata deprecation notice
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

def parse_args():
    parser = argparse.ArgumentParser(description="Train SDV GaussianCopula and sample synthetic rows.")
    parser.add_argument("--real", required=True, help="Path to input CSV with real data.")
    parser.add_argument("--rows", type=int, default=100, help="Number of synthetic rows to generate (default: 100).")
    parser.add_argument("--id", dest="pk", default=None,
                        help="Primary key column name. If omitted, auto-detect a single column ending with 'id' (case-insensitive).")
    parser.add_argument("--outdir", default="output", help="Directory to write outputs (default: output).")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the trained model to outdir as sdv-gaussiancopula-<DATE>.pkl.")
    parser.add_argument("--save-metadata", action="store_true",
                        help="Save detected metadata to outdir as metadata.json.")
    return parser.parse_args()

def main():
    args = parse_args()
    t0 = time.time()

    # Load data
    print(f"[INFO] Reading CSV: {args.real}")
    data = pd.read_csv(args.real)
    print(f"[INFO] Loaded shape: {data.shape}")

    # Build metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Determine primary key
    if args.pk:
        pk_col = args.pk
        if pk_col not in data.columns:
            print(f"[ERROR] --id '{pk_col}' not found in columns: {list(data.columns)}", file=sys.stderr)
            sys.exit(1)
    else:
        id_cols = [c for c in data.columns if re.search(r'(?i)id$', str(c))]
        if len(id_cols) != 1:
            print(f"[ERROR] Auto-detect expected exactly 1 id-suffix column, found {id_cols}. Specify --id.", file=sys.stderr)
            sys.exit(1)
        pk_col = id_cols[0]
    print(f"[INFO] Using primary key: {pk_col}")

    # Set primary key (overwrites any existing)
    metadata.set_primary_key(column_name=pk_col)

    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save metadata only if requested
    if args.save_metadata:
        metadata_json = os.path.join(args.outdir, "metadata.json")
        metadata.save_to_json(metadata_json, mode="overwrite")
        print(f"[INFO] Metadata saved to: {metadata_json}")

    # Initialize and fit synthesizer
    print("[INFO] Initializing GaussianCopulaSynthesizer")
    model = GaussianCopulaSynthesizer(metadata)
    print("[INFO] Fitting synthesizer...")
    model.fit(data)
    print("[INFO] Fit complete.")

    # Save model only if requested
    if args.save_model:
        model_path = os.path.join(args.outdir, f"sdv-gaussiancopula-{datetime.now().strftime('%Y%m%d')}.pkl")
        model.save(model_path)
        print(f"[INFO] Model saved to: {model_path}")

    # Sample synthetic rows
    n = int(args.rows)
    print(f"[INFO] Sampling {n} synthetic rows...")
    new_data = model.sample(num_rows=n)

    # Save synthetic CSV with date
    date_str = datetime.now().strftime("%Y%m%d")
    synth_csv = os.path.join(args.outdir, f"synthetic_data_{date_str}.csv")
    new_data.to_csv(synth_csv, index=False)
    print(f"[INFO] Synthetic data saved to: {synth_csv}")

    print(f"[INFO] Done in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()