"""
02_data_cleaning.py
Data cleaning for Online Retail II dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
XLSX_FILE = DATA_RAW_DIR / "online_retail_II.xlsx"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

def load_data():
    """Load both sheets and merge"""
    print("[INFO] Loading data from Excel file...")
    xls = pd.ExcelFile(XLSX_FILE)

    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_FILE, sheet_name=sheet)
        print(f"  Loaded {sheet}: {df.shape[0]:,} rows")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[INFO] Total records after merge: {df.shape[0]:,}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Standardize column names
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    if 'Customer ID' in df.columns:
        df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

    return df

def clean_data(df):
    """Clean the dataset"""
    print("\n" + "="*70)
    print("[STEP 1] DATA CLEANING")
    print("="*70)

    initial_rows = len(df)

    # Step 1: Remove missing CustomerID
    print("\n[1.1] Remove missing CustomerID...")
    before = len(df)
    df = df.dropna(subset=['CustomerID'])
    removed = before - len(df)
    print(f"  Removed: {removed:,} rows | Remaining: {len(df):,}")

    # Step 2: Remove invalid Quantity (<=0) for normal invoices
    print("\n[1.2] Remove invalid Quantity (<=0) and Price (<=0)...")
    before = len(df)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    removed = before - len(df)
    print(f"  Removed: {removed:,} rows | Remaining: {len(df):,}")

    # Step 3: Create TotalPrice
    print("\n[1.3] Create TotalPrice column...")
    df['TotalPrice'] = df['Quantity'] * df['Price']
    print(f"  Created TotalPrice = Quantity x Price")

    # Step 4: Parse InvoiceDate
    print("\n[1.4] Parse InvoiceDate to datetime...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"  Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # Step 5: Convert CustomerID to int
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Summary
    total_removed = initial_rows - len(df)
    print("\n" + "-"*70)
    print(f"Initial rows: {initial_rows:,}")
    print(f"Final rows: {len(df):,}")
    print(f"Total removed: {total_removed:,} ({100*total_removed/initial_rows:.1f}%)")
    print("-"*70)

    return df

def generate_report(df):
    """Generate cleaning report"""
    print("\n" + "="*70)
    print("[STEP 2] DATASET STATISTICS")
    print("="*70)

    print(f"\nDate range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    print(f"Total transactions: {len(df):,}")
    print(f"Total customers: {df['CustomerID'].nunique():,}")
    print(f"Total unique invoices: {df['Invoice'].nunique():,}")
    print(f"Total unique products: {df['StockCode'].nunique():,}")

    print(f"\nCountries: {df['Country'].nunique():,}")
    print(f"  Top 10 countries:")
    top_countries = df.groupby('Country').size().sort_values(ascending=False).head(10)
    for country, count in top_countries.items():
        print(f"    {country}: {count:,}")

    print(f"\nRevenue statistics:")
    print(f"  Total revenue: ${df['TotalPrice'].sum():,.2f}")
    print(f"  Mean transaction: ${df['TotalPrice'].mean():,.2f}")
    print(f"  Median transaction: ${df['TotalPrice'].median():,.2f}")
    print(f"  Max transaction: ${df['TotalPrice'].max():,.2f}")

    print(f"\nQuantity statistics:")
    print(f"  Mean quantity: {df['Quantity'].mean():.2f}")
    print(f"  Max quantity: {df['Quantity'].max()}")

    print(f"\nCustomer statistics:")
    cust_stats = df.groupby('CustomerID').agg({
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'Invoice': 'NumInvoices', 'TotalPrice': 'TotalSpent'})
    print(f"  Mean purchases per customer: {cust_stats['NumInvoices'].mean():.2f}")
    print(f"  Max purchases per customer: {cust_stats['NumInvoices'].max()}")
    print(f"  Mean spending per customer: ${cust_stats['TotalSpent'].mean():,.2f}")
    print(f"  Max spending per customer: ${cust_stats['TotalSpent'].max():,.2f}")

    print(f"\nMissing values after cleaning:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing:,}")

    return {
        'n_rows': len(df),
        'n_customers': df['CustomerID'].nunique(),
        'n_invoices': df['Invoice'].nunique(),
        'n_products': df['StockCode'].nunique(),
        'date_min': df['InvoiceDate'].min(),
        'date_max': df['InvoiceDate'].max(),
        'total_revenue': df['TotalPrice'].sum()
    }

def main():
    print("\n" + "="*70)
    print("ONLINE RETAIL II - DATA CLEANING PIPELINE")
    print("="*70)

    # Load
    df = load_data()

    # Clean
    df = clean_data(df)

    # Report
    stats = generate_report(df)

    # Save
    print("\n" + "="*70)
    print("[STEP 3] SAVE CLEANED DATA")
    print("="*70)
    df.to_csv(CLEANED_FILE, index=False)
    print(f"\n[OK] Cleaned data saved to: {CLEANED_FILE}")
    print(f"     Rows: {len(df):,} | Columns: {len(df.columns)}")

    # Save metadata
    metadata = {
        'original_rows': 1521251 + 1080660,  # approximate from both sheets
        'cleaned_rows': stats['n_rows'],
        'n_customers': stats['n_customers'],
        'n_invoices': stats['n_invoices'],
        'n_products': stats['n_products'],
        'date_min': str(stats['date_min']),
        'date_max': str(stats['date_max']),
        'total_revenue': stats['total_revenue']
    }

    import json
    with open(DATA_PROCESSED_DIR / "cleaning_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Metadata saved")
    print("\n" + "="*70)
    print("[DONE] Data cleaning completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
