"""
03_eda.py
Exploratory Data Analysis for Online Retail II
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def load_data():
    """Load cleaned data"""
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def eda_summary(df):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("[EDA] DATASET SUMMARY")
    print("="*70)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nDate range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")

    print(f"\n--- CUSTOMERS ---")
    print(f"Total unique customers: {df['CustomerID'].nunique():,}")
    print(f"Mean transactions per customer: {df.groupby('CustomerID')['Invoice'].nunique().mean():.2f}")
    print(f"Median transactions per customer: {df.groupby('CustomerID')['Invoice'].nunique().median():.0f}")

    print(f"\n--- TRANSACTIONS ---")
    print(f"Total transactions: {len(df):,}")
    print(f"Unique invoices: {df['Invoice'].nunique():,}")
    print(f"Unique products: {df['StockCode'].nunique():,}")
    print(f"Mean items per transaction: {df.groupby('Invoice')['Quantity'].sum().mean():.2f}")

    print(f"\n--- REVENUE ---")
    print(f"Total revenue: ${df['TotalPrice'].sum():,.2f}")
    print(f"Mean revenue per transaction: ${df.groupby('Invoice')['TotalPrice'].sum().mean():,.2f}")
    print(f"Median revenue per transaction: ${df.groupby('Invoice')['TotalPrice'].sum().median():,.2f}")

    print(f"\n--- GEOGRAPHY ---")
    print(f"Countries: {df['Country'].nunique()}")
    top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(5)
    for country, revenue in top_countries.items():
        pct = 100 * revenue / df['TotalPrice'].sum()
        print(f"  {country}: ${revenue:,.0f} ({pct:.1f}%)")

def plot_monthly_revenue(df):
    """Plot monthly revenue trend"""
    print("\n[Plot] Monthly revenue trend...")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_revenue = df.groupby('YearMonth')['TotalPrice'].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_revenue.plot(ax=ax, color='#2E86AB', linewidth=2)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Revenue ($)', fontsize=11)
    ax.set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_monthly_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_monthly_revenue.png")

def plot_customer_distribution(df):
    """Plot customer distribution by spending"""
    print("[Plot] Customer spending distribution...")

    customer_spending = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(customer_spending, bins=50, color='#A23B72', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Total Spending ($)', fontsize=11)
    axes[0].set_ylabel('Number of Customers', fontsize=11)
    axes[0].set_title('Distribution of Customer Spending', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Log scale histogram
    axes[1].hist(customer_spending, bins=50, color='#F18F01', edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Total Spending ($)', fontsize=11)
    axes[1].set_ylabel('Number of Customers (log)', fontsize=11)
    axes[1].set_title('Distribution of Customer Spending (Log Scale)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_customer_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_customer_distribution.png")

def plot_rfm_segments(df):
    """Compute and plot RFM segments"""
    print("[Plot] RFM segment distribution...")

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }).reset_index()

    # Create quartile-based segments for visualization
    rfm['R_Quartile'] = pd.qcut(rfm['Recency'], q=4, labels=['1 (Recent)', '2', '3', '4 (Old)'], duplicates='drop')
    rfm['F_Quartile'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=['1 (Low)', '2', '3', '4 (High)'], duplicates='drop')
    rfm['M_Quartile'] = pd.qcut(rfm['Monetary'], q=4, labels=['1 (Low)', '2', '3', '4 (High)'], duplicates='drop')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Recency
    r_counts = rfm['R_Quartile'].value_counts().sort_index()
    axes[0].bar(range(len(r_counts)), r_counts.values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[0].set_xticks(range(len(r_counts)))
    axes[0].set_xticklabels(r_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Customers', fontsize=11)
    axes[0].set_title('Recency Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Frequency
    f_counts = rfm['F_Quartile'].value_counts().sort_index()
    axes[1].bar(range(len(f_counts)), f_counts.values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[1].set_xticks(range(len(f_counts)))
    axes[1].set_xticklabels(f_counts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Customers', fontsize=11)
    axes[1].set_title('Frequency Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Monetary
    m_counts = rfm['M_Quartile'].value_counts().sort_index()
    axes[2].bar(range(len(m_counts)), m_counts.values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[2].set_xticks(range(len(m_counts)))
    axes[2].set_xticklabels(m_counts.index, rotation=45, ha='right')
    axes[2].set_ylabel('Number of Customers', fontsize=11)
    axes[2].set_title('Monetary Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_rfm_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_rfm_distribution.png")

    return rfm

def plot_country_revenue(df):
    """Plot top countries by revenue"""
    print("[Plot] Country revenue...")

    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    country_revenue.plot(kind='barh', ax=ax, color='#2E86AB')
    ax.set_xlabel('Revenue ($)', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)
    ax.set_title('Top 10 Countries by Revenue', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_top_countries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eda_top_countries.png")

def create_eda_table(df, rfm):
    """Create Table 1 for paper"""
    print("[Table] Dataset statistics table...")

    table_data = {
        'Metric': [
            'Date Range',
            'Total Transactions',
            'Unique Invoices',
            'Unique Customers',
            'Unique Products',
            'Unique Countries',
            'Total Revenue',
            'Mean Transaction Value',
            'Median Transaction Value',
            'Mean Purchases/Customer',
            'Median Purchases/Customer',
            'Mean Spending/Customer',
            'Median Spending/Customer'
        ],
        'Value': [
            f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}",
            f"{len(df):,}",
            f"{df['Invoice'].nunique():,}",
            f"{df['CustomerID'].nunique():,}",
            f"{df['StockCode'].nunique():,}",
            f"{df['Country'].nunique()}",
            f"${df['TotalPrice'].sum():,.2f}",
            f"${df.groupby('Invoice')['TotalPrice'].sum().mean():,.2f}",
            f"${df.groupby('Invoice')['TotalPrice'].sum().median():,.2f}",
            f"{rfm['Frequency'].mean():.2f}",
            f"{rfm['Frequency'].median():.0f}",
            f"${rfm['Monetary'].mean():,.2f}",
            f"${rfm['Monetary'].median():,.2f}"
        ]
    }

    table_df = pd.DataFrame(table_data)
    table_df.to_csv(DATA_PROCESSED_DIR / 'table_1_dataset_statistics.csv', index=False)
    print("  Saved: table_1_dataset_statistics.csv")

    print("\n" + "-"*70)
    print("TABLE 1: Dataset Statistics")
    print("-"*70)
    print(table_df.to_string(index=False))
    print("-"*70)

def main():
    print("\n" + "="*70)
    print("[EDA] EXPLORATORY DATA ANALYSIS")
    print("="*70)

    df = load_data()
    eda_summary(df)

    print("\n" + "="*70)
    print("[STEP 1] GENERATE PLOTS")
    print("="*70)

    plot_monthly_revenue(df)
    plot_customer_distribution(df)
    rfm = plot_rfm_segments(df)
    plot_country_revenue(df)

    print("\n" + "="*70)
    print("[STEP 2] CREATE TABLES")
    print("="*70)

    create_eda_table(df, rfm)

    print("\n" + "="*70)
    print("[DONE] EDA completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
