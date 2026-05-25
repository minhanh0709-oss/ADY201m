"""
01_data_download.py
Download and extract Online Retail II dataset
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Set UTF-8 encoding for Windows
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Paths
DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
ZIP_FILE = DATA_RAW_DIR / "online_retail_ii.zip"
XLSX_FILE = DATA_RAW_DIR / "online_retail_II.xlsx"

def extract_dataset():
    """Extract ZIP file if it exists"""
    if ZIP_FILE.exists():
        print(f"[OK] Found ZIP file: {ZIP_FILE}")
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW_DIR)
            print(f"[OK] Extracted to {DATA_RAW_DIR}")
        except Exception as e:
            print(f"[ERROR] Error extracting ZIP: {e}")
            return False
    else:
        print(f"[ERROR] ZIP file not found: {ZIP_FILE}")
        print("  Please download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii")
        print("  Or: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        return False
    return True

def verify_dataset():
    """Verify dataset exists and can be read"""
    if not XLSX_FILE.exists():
        # Try to find any .xlsx file in raw directory
        xlsx_files = list(DATA_RAW_DIR.glob("*.xlsx"))
        if xlsx_files:
            xlsx_path = xlsx_files[0]
            print(f"  Found: {xlsx_path.name}")
        else:
            print(f"[ERROR] No Excel file found in {DATA_RAW_DIR}")
            return False
    else:
        xlsx_path = XLSX_FILE

    try:
        # Read both sheets to verify
        xls = pd.ExcelFile(xlsx_path)
        print(f"\n[OK] Excel file readable")
        print(f"  Sheets: {xls.sheet_names}")

        # Load first few rows of each sheet
        for sheet in xls.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=5)
            print(f"\n  Sheet '{sheet}': {df.shape[0]} rows (preview)")
            print(f"    Columns: {list(df.columns)[:5]}...")

        return True
    except Exception as e:
        print(f"[ERROR] Error reading Excel file: {e}")
        return False

def main():
    print("=" * 60)
    print("Step 1: Extract Online Retail II Dataset")
    print("=" * 60)

    # Extract
    if not extract_dataset():
        return False

    # Verify
    print("\nStep 2: Verify Dataset")
    print("-" * 60)
    if not verify_dataset():
        return False

    print("\n" + "=" * 60)
    print("[OK] Dataset ready for processing!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
