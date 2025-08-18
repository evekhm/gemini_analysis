import json
import warnings
from datetime import datetime, timezone, timedelta
import argparse

import google.auth
import google.auth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
import os

warnings.filterwarnings('ignore')

# --- USER INPUT SECTION ---
# BigQuery dataset and table IDs
dataset_id = os.getenv("DATASET", "MY_DATASET")
gemini_table_id = os.getenv("GEMINI_LOG_TABLE", "gemini_flash_logs")
project_id = os.getenv("PROJECT_ID")
# --- END USER INPUT SECTION ---

if not project_id:
    # Initialize the BigQuery client
    _, project_id = google.auth.default()

client = bigquery.Client(project=project_id)

def parse_arguments():
    """Parse command line arguments for start and end timestamps"""
    parser = argparse.ArgumentParser(description='Analyze Gemini log data with customizable time range')

    # Default start time: 90 days ago
    default_start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    # Default end time: current UTC time
    default_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    parser.add_argument('--start', '-s',
                        default=None,
                        help=f'Start timestamp in format "YYYY-MM-DD HH:MM:SS" (default: {default_start} if no -d option)')

    parser.add_argument('--end', '-e',
                        default=default_end,
                        help=f'End timestamp in format "YYYY-MM-DD HH:MM:SS" (default: current UTC time)')

    parser.add_argument('--days', '-d',
                        type=int,
                        default=None,
                        help='Number of days back from now to analyze (e.g., -d 10 for last 10 days). Overrides --start if specified.')

    return parser.parse_args()

def validate_timestamp(timestamp_str):
    """Validate timestamp format"""
    try:
        datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False

# Parse command line arguments
args = parse_arguments()

# Handle days option
if args.days is not None:
    if args.days <= 0:
        print(f"Error: Days must be a positive integer, got: {args.days}")
        exit(1)

    # Calculate start time based on days back from now
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    start_filter_timestamp = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_filter_timestamp = args.end

    print(f"Using -d {args.days}: analyzing last {args.days} days")

    # If end was also specified, use it
    if args.end != datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"):
        if not validate_timestamp(args.end):
            print(f"Error: Invalid end timestamp format: {args.end}")
            print("Please use format: YYYY-MM-DD HH:MM:SS")
            exit(1)
        end_filter_timestamp = args.end
        end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        start_dt = end_dt - timedelta(days=args.days)
        start_filter_timestamp = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Using custom end time: {args.end}")
        print(f"Calculated start time: {start_filter_timestamp}")
    else:
        end_filter_timestamp = end_dt.strftime("%Y-%m-%d %H:%M:%S")
else:
    # Use start/end arguments or defaults
    default_start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    start_filter_timestamp = args.start if args.start is not None else default_start
    end_filter_timestamp = args.end

    # Validate timestamps
    if not validate_timestamp(start_filter_timestamp):
        print(f"Error: Invalid start timestamp format: {start_filter_timestamp}")
        print("Please use format: YYYY-MM-DD HH:MM:SS")
        exit(1)

    if not validate_timestamp(end_filter_timestamp):
        print(f"Error: Invalid end timestamp format: {end_filter_timestamp}")
        print("Please use format: YYYY-MM-DD HH:MM:SS")
        exit(1)

    # Check that start is before end
    start_dt = datetime.strptime(start_filter_timestamp, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_filter_timestamp, "%Y-%m-%d %H:%M:%S")

    if start_dt >= end_dt:
        print(f"Error: Start time ({start_filter_timestamp}) must be before end time ({end_filter_timestamp})")
        exit(1)



# Create filename-safe timestamp strings
start_filter_timestamp_str = start_filter_timestamp.replace(" ","_").replace(":","-")
end_filter_timestamp_str = end_filter_timestamp.replace(" ","_").replace(":","-")
timestamp_str = f"{start_filter_timestamp_str}_to_{end_filter_timestamp_str}"

script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)


print(f"Using dataset_id={dataset_id}, gemini_table_id={gemini_table_id}, project_id={project_id}")
print(f"Analyzing data from {start_filter_timestamp} to {end_filter_timestamp} UTC")
print(f"Time range: {(end_dt - start_dt).days} days, {(end_dt - start_dt).seconds // 3600} hours")

# Define the SQL query to get latency, logging time, full_response, and model
# MODIFIED: Extract only the JSON string from full_response to avoid schema issues
gemini_sql = f"""
  SELECT
    CAST(JSON_EXTRACT_SCALAR(metadata, '$.request_latency') AS FLOAT64) / 1000 AS latency_seconds,
    logging_time,
    TO_JSON_STRING(full_response) as full_response_json,
    model
  FROM
    `{project_id}.{dataset_id}.{gemini_table_id}`
  WHERE
    logging_time BETWEEN '{start_filter_timestamp}' AND '{end_filter_timestamp}'
    AND full_response IS NOT NULL
    AND model IS NOT NULL
  ORDER BY logging_time
"""

def extract_token_count(full_response_str):
    """Extract totalTokenCount from full_response JSON string"""
    try:
        if pd.isna(full_response_str) or full_response_str == '':
            return None

        # Parse JSON
        response_json = json.loads(full_response_str)

        # Extract totalTokenCount
        usage_metadata = response_json.get('usageMetadata', {})
        total_token_count = usage_metadata.get('totalTokenCount')

        return total_token_count
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None

def extract_model_name(model_path):
    """Extract model name from model path like 'publishers/google/models/gemini-2.0-flash-lite'"""
    try:
        if pd.isna(model_path) or model_path == '':
            return None

        # Split by '/' and get the part after 'models/'
        parts = model_path.split('/')
        if 'models' in parts:
            model_index = parts.index('models')
            if model_index + 1 < len(parts):
                return parts[model_index + 1]

        return model_path  # Return original if parsing fails
    except Exception as e:
        return None

def safe_polyfit(x, y, degree=1):
    """Safely perform polynomial fitting with error handling"""
    try:
        if len(x) < 2 or len(y) < 2:
            return None, None

        # Check for constant values
        if np.std(x) == 0 or np.std(y) == 0:
            return None, None

        # Remove any NaN or infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return None, None

        x_clean = np.array(x)[mask]
        y_clean = np.array(y)[mask]

        z = np.polyfit(x_clean, y_clean, degree)
        p = np.poly1d(z)
        return z, p
    except (np.linalg.LinAlgError, ValueError, TypeError) as e:
        print(f"Warning: Could not fit polynomial trend line: {e}")
        return None, None

def analyze_model_data(df_model, model_name):
    """Generate comprehensive analysis for a specific model"""

    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR MODEL: {model_name}")
    print(f"{'='*60}")

    if df_model.empty:
        print("No data found for this model.")
        return

    # === DIAGNOSTIC CODE ===
    print(f"\n--- DIAGNOSTIC: Actual Time Distribution ---")
    print("Hour distribution in your data:")
    hour_counts = df_model['logging_time'].dt.hour.value_counts().sort_index()
    print(hour_counts)

    print(f"\nTime range: {df_model['logging_time'].min()} to {df_model['logging_time'].max()}")

    # Check if there's ANY data between 8-15
    morning_afternoon = df_model[(df_model['logging_time'].dt.hour >= 8) &
                                 (df_model['logging_time'].dt.hour <= 15)]
    print(f"\nRequests between 8:00-15:59: {len(morning_afternoon)} out of {len(df_model)} total")

    if len(morning_afternoon) > 0:
        print("Sample of 8-15 hour data:")
        print(morning_afternoon[['logging_time', 'latency_seconds']].head())
    else:
        print("NO DATA found between 8:00-15:59!")
    print("=" * 50)

    # Create latency categories for pattern analysis - UPDATED CATEGORIES
    def categorize_latency(latency):
        if latency < 1.0:
            return 'Fast (< 1s)'
        elif latency < 2.0:
            return 'Medium (1-2s)'
        elif latency < 3.0:
            return 'Slow (2-3s)'
        elif latency < 5.0:
            return 'Very Slow (3-5s)'
        else:
            return 'Outliers (5s+)'

    df_model['latency_category'] = df_model['latency_seconds'].apply(categorize_latency)

    # Round latency for grouping
    df_model['latency_rounded'] = df_model['latency_seconds'].round(1)

    # Extract time components for temporal analysis
    df_model['hour'] = df_model['logging_time'].dt.hour
    df_model['minute'] = df_model['logging_time'].dt.minute
    df_model['date'] = df_model['logging_time'].dt.date

    # === FIX MIDNIGHT BOUNDARY ISSUE ===
    time_range = df_model['logging_time'].max() - df_model['logging_time'].min()
    if time_range.total_seconds() < 24 * 3600:  # Less than 24 hours
        has_late_night = any(df_model['hour'] >= 22)
        has_early_morning = any(df_model['hour'] <= 6)

        if has_late_night and has_early_morning:
            print("Detected midnight boundary crossing - adjusting hours for continuous visualization")
            df_model['adjusted_hour'] = df_model['hour'].copy()
            df_model.loc[df_model['hour'] <= 6, 'adjusted_hour'] = df_model.loc[df_model['hour'] <= 6, 'hour'] + 24
            use_adjusted_hours = True
        else:
            df_model['adjusted_hour'] = df_model['hour']
            use_adjusted_hours = False
    else:
        df_model['adjusted_hour'] = df_model['hour']
        use_adjusted_hours = False

    # === CALCULATE STANDARD DEVIATION STATISTICS ===
    mean_latency = df_model['latency_seconds'].mean()
    std_latency = df_model['latency_seconds'].std()
    count_total = len(df_model)

    # Define outlier thresholds
    std_2_threshold = mean_latency + 2 * std_latency
    std_3_threshold = mean_latency + 3 * std_latency

    # Count requests greater than 2 and 3 STD deviations from the mean
    count_gt_2_std = len(df_model[df_model['latency_seconds'] > std_2_threshold])
    count_gt_3_std = len(df_model[df_model['latency_seconds'] > std_3_threshold])

    # Calculate percentages
    percent_gt_2_std = (count_gt_2_std / count_total) * 100 if count_total > 0 else 0
    percent_gt_3_std = (count_gt_3_std / count_total) * 100 if count_total > 0 else 0

    # Print basic statistics
    print(f"\n--- Data Summary for {model_name} ---")
    print(f"Total requests: {len(df_model)}")
    print(f"Date range: {df_model['logging_time'].min()} to {df_model['logging_time'].max()}")
    print(f"Latency statistics:")
    print(f"  Mean: {df_model['latency_seconds'].mean():.3f}s")
    print(f"  Std Dev: {std_latency:.3f}s")
    print(f"  Median: {df_model['latency_seconds'].median():.3f}s")
    print(f"  95th percentile: {df_model['latency_seconds'].quantile(0.95):.3f}s")
    print(f"  99th percentile: {df_model['latency_seconds'].quantile(0.99):.3f}s")
    print(f"  Max: {df_model['latency_seconds'].max():.3f}s")
    print(f"  > 2 STD ({std_2_threshold:.3f}s): {count_gt_2_std} ({percent_gt_2_std:.2f}%)")
    print(f"  > 3 STD ({std_3_threshold:.3f}s): {count_gt_3_std} ({percent_gt_3_std:.2f}%)")

    # Token statistics
    df_tokens = None
    if 'total_token_count' in df_model.columns and df_model['total_token_count'].notna().any():
        df_tokens = df_model.dropna(subset=['total_token_count'])
        if len(df_tokens) > 0:
            print(f"Token statistics:")
            print(f"  Mean tokens: {df_tokens['total_token_count'].mean():.1f}")
            print(f"  Median tokens: {df_tokens['total_token_count'].median():.1f}")
            print(f"  Min tokens: {df_tokens['total_token_count'].min()}")
            print(f"  Max tokens: {df_tokens['total_token_count'].max()}")

    print(f"\n--- Latency Distribution ---")
    print(df_model['latency_category'].value_counts().sort_index())

    # Calculate key metrics for tables - UPDATED FOR NEW CATEGORIES
    fast_pct = len(df_model[df_model['latency_seconds'] < 1.0]) / len(df_model) * 100
    slow_pct = len(df_model[df_model['latency_seconds'] > 3.0]) / len(df_model) * 100
    outlier_pct = len(df_model[df_model['latency_seconds'] > 5.0]) / len(df_model) * 100

    # Token-latency correlation
    token_latency_corr = None
    correlation_strength = "N/A"
    if df_tokens is not None and len(df_tokens) > 1:
        token_latency_corr = np.corrcoef(df_tokens['total_token_count'], df_tokens['latency_seconds'])[0, 1]
        if abs(token_latency_corr) > 0.7:
            correlation_strength = "Strong"
        elif abs(token_latency_corr) > 0.3:
            correlation_strength = "Moderate"
        else:
            correlation_strength = "Weak"

    # Create comprehensive visualizations
    plt.style.use('default')
    fig = plt.figure(figsize=(32, 28))
    fig.suptitle(f'Gemini Log Analysis - {model_name}\n{start_filter_timestamp} to {end_filter_timestamp} UTC',
                 fontsize=18, fontweight='bold', y=0.98)

    # === ROW 1: SUMMARY TABLES (3 tables across the full width) ===

    # Table 1: Basic Statistics (ENHANCED WITH STD DEV DATA)
    ax1 = plt.subplot(4, 3, 1)
    ax1.axis('tight')
    ax1.axis('off')

    basic_stats = [
        ['Metric', 'Value'],
        ['Total Requests', f'{len(df_model):,}'],
        ['Date Range', f'{df_model["logging_time"].min().strftime("%Y-%m-%d %H:%M")} to {df_model["logging_time"].max().strftime("%Y-%m-%d %H:%M")}'],
        ['Mean Latency', f'{mean_latency:.3f}s'],
        ['Std Deviation', f'{std_latency:.3f}s'],
        ['Median Latency', f'{df_model["latency_seconds"].median():.3f}s'],
        ['P95 Latency', f'{df_model["latency_seconds"].quantile(0.95):.3f}s'],
        ['P99 Latency', f'{df_model["latency_seconds"].quantile(0.99):.3f}s'],
        ['Max Latency', f'{df_model["latency_seconds"].max():.3f}s'],
        ['> 2 STD', f'{count_gt_2_std} ({percent_gt_2_std:.2f}%)'],
        ['> 3 STD', f'{count_gt_3_std} ({percent_gt_3_std:.2f}%)'],
    ]

    table1 = ax1.table(cellText=basic_stats[1:], colLabels=basic_stats[0],
                       cellLoc='left', loc='center', colWidths=[0.4, 0.6])
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 1.8)

    # Style the header
    for i in range(len(basic_stats[0])):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight the STD deviation rows
    for row_idx in [9, 10]:  # > 2 STD and > 3 STD rows
        for col_idx in range(len(basic_stats[0])):
            table1[(row_idx, col_idx)].set_facecolor('#FFE0B2')

    plt.title('Basic Statistics', fontsize=16, fontweight='bold', pad=20)

    # Table 2: Token Statistics
    ax2 = plt.subplot(4, 3, 2)
    ax2.axis('tight')
    ax2.axis('off')

    if df_tokens is not None and len(df_tokens) > 0:
        token_stats = [
            ['Metric', 'Value'],
            ['Mean Tokens', f'{df_tokens["total_token_count"].mean():.1f}'],
            ['Median Tokens', f'{df_tokens["total_token_count"].median():.1f}'],
            ['Min Tokens', f'{df_tokens["total_token_count"].min():,}'],
            ['Max Tokens', f'{df_tokens["total_token_count"].max():,}'],
            ['Token-Latency Corr', f'{token_latency_corr:.3f}' if token_latency_corr is not None else 'N/A'],
            ['Correlation Strength', correlation_strength],
        ]
    else:
        token_stats = [
            ['Metric', 'Value'],
            ['Mean Tokens', 'N/A'],
            ['Median Tokens', 'N/A'],
            ['Min Tokens', 'N/A'],
            ['Max Tokens', 'N/A'],
            ['Token-Latency Corr', 'N/A'],
            ['Correlation Strength', 'N/A'],
        ]

    table2 = ax2.table(cellText=token_stats[1:], colLabels=token_stats[0],
                       cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2.0)

    # Style the header
    for i in range(len(token_stats[0])):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Token Statistics', fontsize=16, fontweight='bold', pad=20)

    # Table 3: Performance Distribution - UPDATED FOR NEW CATEGORIES
    ax3 = plt.subplot(4, 3, 3)
    ax3.axis('tight')
    ax3.axis('off')

    perf_stats = [
        ['Category', 'Count', 'Percentage'],
        ['Fast (< 1s)', f'{len(df_model[df_model["latency_seconds"] < 1.0]):,}', f'{fast_pct:.1f}%'],
        ['Medium (1-2s)', f'{len(df_model[(df_model["latency_seconds"] >= 1.0) & (df_model["latency_seconds"] < 2.0)]):,}', f'{len(df_model[(df_model["latency_seconds"] >= 1.0) & (df_model["latency_seconds"] < 2.0)])/len(df_model)*100:.1f}%'],
        ['Slow (2-3s)', f'{len(df_model[(df_model["latency_seconds"] >= 2.0) & (df_model["latency_seconds"] < 3.0)]):,}', f'{len(df_model[(df_model["latency_seconds"] >= 2.0) & (df_model["latency_seconds"] < 3.0)])/len(df_model)*100:.1f}%'],
        ['Very Slow (3-5s)', f'{len(df_model[(df_model["latency_seconds"] >= 3.0) & (df_model["latency_seconds"] < 5.0)]):,}', f'{len(df_model[(df_model["latency_seconds"] >= 3.0) & (df_model["latency_seconds"] < 5.0)])/len(df_model)*100:.1f}%'],
        ['Outliers (5s+)', f'{len(df_model[df_model["latency_seconds"] >= 5.0]):,}', f'{outlier_pct:.1f}%'],
    ]

    table3 = ax3.table(cellText=perf_stats[1:], colLabels=perf_stats[0],
                       cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1, 2.0)

    # Style the header
    for i in range(len(perf_stats[0])):
        table3[(0, i)].set_facecolor('#FF9800')
        table3[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Performance Distribution', fontsize=16, fontweight='bold', pad=20)

    # === ROW 2: TIME SERIES AND TOKEN ANALYSIS ===

    # Plot 1: Detailed histogram (ENHANCED WITH STD DEV LINES)
    ax1 = plt.subplot(4, 3, 4)
    plt.hist(df_model['latency_seconds'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Detailed Latency Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Frequency')

    # Add mean and standard deviation lines
    plt.axvline(mean_latency, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_latency:.2f}s')
    plt.axvline(mean_latency + std_latency, color='green', linestyle=':', linewidth=2,
                label=f'1 STD: {std_latency:.2f}s')
    plt.axvline(mean_latency - std_latency, color='green', linestyle=':', linewidth=2)
    plt.axvline(std_2_threshold, color='orange', linestyle='-.', linewidth=2,
                label=f'2 STD: {std_2_threshold:.2f}s')
    plt.axvline(std_3_threshold, color='purple', linestyle='-.', linewidth=2,
                label=f'3 STD: {std_3_threshold:.2f}s')

    plt.legend(fontsize=10)

    # Plot 2: Histogram with categories - UPDATED FOR NEW CATEGORIES
    ax6 = plt.subplot(4, 3, 5)
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    category_order = ['Fast (< 1s)', 'Medium (1-2s)', 'Slow (2-3s)', 'Very Slow (3-5s)', 'Outliers (5s+)']
    category_counts = df_model['latency_category'].value_counts().reindex(category_order, fill_value=0)

    bars = plt.bar(range(len(category_counts)), category_counts.values, color=colors)
    plt.title('Latency Distribution by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Latency Category')
    plt.ylabel('Count')
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')

    for bar, count in zip(bars, category_counts.values):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(count), ha='center', va='bottom')

    # Plot 3: Latency vs Token Count
    ax5 = plt.subplot(4, 3, 6)
    if df_tokens is not None and len(df_tokens) > 0:
        scatter = plt.scatter(df_tokens['total_token_count'], df_tokens['latency_seconds'],
                              alpha=0.6, s=40, c=df_tokens['latency_seconds'],
                              cmap='viridis', edgecolors='black', linewidth=0.5)

        plt.colorbar(scatter, label='Latency (seconds)')
        plt.title('Latency vs Token Count', fontsize=14, fontweight='bold')
        plt.xlabel('Total Token Count')
        plt.ylabel('Latency (seconds)')
        plt.grid(True, alpha=0.3)

        if token_latency_corr is not None:
            plt.text(0.05, 0.95, f'Correlation: {token_latency_corr:.3f}',
                     transform=ax5.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Safe trend line
            z, p = safe_polyfit(df_tokens['total_token_count'], df_tokens['latency_seconds'])
            if z is not None and p is not None:
                plt.plot(df_tokens['total_token_count'], p(df_tokens['total_token_count']),
                         "r--", alpha=0.8, linewidth=2)
    else:
        plt.text(0.5, 0.5, 'No token count data available',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        plt.title('Latency vs Token Count', fontsize=14, fontweight='bold')


    # Plot 7: Cumulative distribution
    ax10 = plt.subplot(4, 3, 7)
    sorted_latencies = np.sort(df_model['latency_seconds'])
    cumulative_pct = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies) * 100

    plt.plot(sorted_latencies, cumulative_pct, linewidth=2, color='purple')
    plt.title('Cumulative Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Cumulative Percentage')
    plt.grid(True, alpha=0.3)

    for pct in [50, 90, 95, 99]:
        value = np.percentile(df_model['latency_seconds'], pct)
        plt.axvline(value, color='red', linestyle='--', alpha=0.7)
        plt.text(value, pct, f'P{pct}: {value:.2f}s', rotation=90, va='bottom')


    # === ROW 3: DETAILED ANALYSIS ===



    # Plot 5: Load Test Sequence
    ax8 = plt.subplot(4, 3, 8)
    df_sorted = df_model.sort_values('logging_time').reset_index(drop=True)
    df_sorted['request_number'] = range(1, len(df_sorted) + 1)

    plt.scatter(df_sorted['request_number'], df_sorted['latency_seconds'],
                alpha=0.6, s=15, c=df_sorted['latency_seconds'], cmap='viridis')
    plt.plot(df_sorted['request_number'], df_sorted['latency_seconds'],
             color='red', alpha=0.4, linewidth=0.8)

    window_size = max(10, len(df_sorted) // 30)
    if window_size > 1:
        moving_avg = df_sorted['latency_seconds'].rolling(window=window_size, center=True).mean()
        plt.plot(df_sorted['request_number'], moving_avg,
                 color='blue', linewidth=2, alpha=0.8)

    plt.title('Load Test Sequence\n(Request Order vs Latency)', fontsize=14, fontweight='bold')
    plt.xlabel('Request Number (5s intervals)')
    plt.ylabel('Latency (seconds)')
    plt.grid(True, alpha=0.3)

    plt.text(0.02, 0.98, f'Total: {len(df_sorted)} requests\nDuration: ~{len(df_sorted)*5/60:.1f} minutes',
             transform=ax8.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 6: Token Count Distribution
    ax9 = plt.subplot(4, 3, 9)
    if df_tokens is not None and len(df_tokens) > 0:
        plt.hist(df_tokens['total_token_count'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Token Count Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Total Token Count')
        plt.ylabel('Frequency')
        plt.axvline(df_tokens['total_token_count'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df_tokens["total_token_count"].mean():.1f}')
        plt.axvline(df_tokens['total_token_count'].median(), color='green', linestyle='--',
                    label=f'Median: {df_tokens["total_token_count"].median():.1f}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No token count data', ha='center', va='center', transform=ax9.transAxes)
        plt.title('Token Count Distribution', fontsize=14, fontweight='bold')

    # === ROW 4: CUMULATIVE AND TREND ANALYSIS ===


    # 9. Box plot by hour
    ax12 = plt.subplot(4, 3, 11)
    hour_column = 'adjusted_hour' if use_adjusted_hours else 'hour'
    unique_hours = sorted(df_model[hour_column].unique())

    hourly_data = []
    hour_labels = []
    for h in unique_hours:
        data = df_model[df_model[hour_column] == h]['latency_seconds'].values
        if len(data) > 0:
            hourly_data.append(data)
            display_hour = h if h <= 23 else h - 24
            hour_labels.append(f"{display_hour:02d}:00")

    if len(hourly_data) > 0:
        plt.boxplot(hourly_data, labels=hour_labels)
        plt.title(f'Latency Distribution by Hour\n({len(unique_hours)} hours with data)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No hourly data available', ha='center', va='center', transform=ax12.transAxes)
        plt.title('Latency Distribution by Hour', fontsize=14, fontweight='bold')


    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=4.0)

    # Save plot to file
    safe_model_name = model_name.replace("-", "_").replace(".", "_")
    filename = os.path.join(plots_dir, f'gemini_analysis_{safe_model_name}_{timestamp_str}')

    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')


    # plt.show()

    # Print insights (ENHANCED WITH STD DEV INSIGHTS) - UPDATED FOR NEW CATEGORIES
    print(f"\n--- Key Insights for {model_name} ---")

    # Reality check
    print(f"0. DATA REALITY CHECK:")
    hour_dist = df_model['hour'].value_counts().sort_index()
    most_active_hour = hour_dist.idxmax()
    most_active_count = hour_dist.max()
    total_requests = len(df_model)

    print(f"   - Most active hour: {most_active_hour:02d}:00 with {most_active_count} requests ({most_active_count/total_requests*100:.1f}%)")
    print(f"   - Hours with data: {sorted(df_model['hour'].unique())}")

    business_hours = df_model[(df_model['hour'] >= 8) & (df_model['hour'] <= 17)]
    print(f"   - Business hours (8-17): {len(business_hours)} requests ({len(business_hours)/total_requests*100:.1f}%)")

    night_hours = df_model[(df_model['hour'] >= 22) | (df_model['hour'] <= 6)]
    print(f"   - Night hours (22-06): {len(night_hours)} requests ({len(night_hours)/total_requests*100:.1f}%)")

    # Performance distribution - UPDATED FOR NEW CATEGORIES
    print(f"1. Performance Distribution:")
    print(f"   - {fast_pct:.1f}% of requests are fast (< 1s)")
    print(f"   - {slow_pct:.1f}% of requests are slow (> 3s)")
    print(f"   - {outlier_pct:.1f}% of requests are outliers (> 5s)")

    # Standard deviation analysis
    print(f"2. Standard Deviation Analysis:")
    print(f"   - Mean latency: {mean_latency:.3f}s")
    print(f"   - Standard deviation: {std_latency:.3f}s")
    print(f"   - Requests > 2 STD ({std_2_threshold:.3f}s): {count_gt_2_std} ({percent_gt_2_std:.2f}%)")
    print(f"   - Requests > 3 STD ({std_3_threshold:.3f}s): {count_gt_3_std} ({percent_gt_3_std:.2f}%)")

    if percent_gt_2_std > 5:
        print(f"   ⚠️  WARNING: High percentage of requests beyond 2 STD - investigate outliers")
    if percent_gt_3_std > 1:
        print(f"   ⚠️  WARNING: Significant percentage of requests beyond 3 STD - potential system issues")

    # Token analysis
    if df_tokens is not None and len(df_tokens) > 1:
        print(f"3. Token-Latency Relationship:")
        print(f"   - Correlation coefficient: {token_latency_corr:.3f}")
        print(f"   - {correlation_strength} correlation detected")

    # Check for trends
    df_sorted = df_model.sort_values('logging_time').reset_index(drop=True)
    df_sorted['request_order'] = range(len(df_sorted))

    try:
        correlation = np.corrcoef(df_sorted['request_order'], df_sorted['latency_seconds'])[0, 1]
        if abs(correlation) > 0.1:
            trend_direction = "increasing" if correlation > 0 else "decreasing"
            print(f"4. Trend detected: Latency is {trend_direction} over time (correlation: {correlation:.3f})")
    except:
        print(f"4. Could not calculate trend correlation")

    # Print detailed category distribution - NEW
    print(f"5. Detailed Category Distribution:")
    for category in ['Fast (< 1s)', 'Medium (1-2s)', 'Slow (2-3s)', 'Very Slow (3-5s)', 'Outliers (5s+)']:
        count = len(df_model[df_model['latency_category'] == category])
        percentage = (count / len(df_model)) * 100
        print(f"   - {category}: {count} requests ({percentage:.1f}%)")

    print(f"Plots saved as:")
    print(f" - {filename}.png")
    print(f" - {filename}.pdf")

try:
    # Run the query and get the result as a DataFrame
    print("Executing BigQuery...")
    df_gemini = client.query(gemini_sql).to_dataframe()

    # Drop rows where latency or logging time is missing
    df_gemini.dropna(subset=['latency_seconds', 'logging_time'], inplace=True)

    if df_gemini.empty:
        print("No data found for the specified time range.")
        exit()

    # Convert logging_time to datetime for analysis
    df_gemini['logging_time'] = pd.to_datetime(df_gemini['logging_time'])

    # Extract token count and model name
    print("Extracting token counts and model names...")
    # MODIFIED: Use the JSON string column instead of the struct
    df_gemini['total_token_count'] = df_gemini['full_response_json'].apply(extract_token_count)
    df_gemini['model_name'] = df_gemini['model'].apply(extract_model_name)

    # Print overall summary
    print(f"\n--- OVERALL SUMMARY ---")
    print(f"Total requests: {len(df_gemini)}")
    print(f"Requests with token data: {df_gemini['total_token_count'].notna().sum()}")
    print(f"Unique models: {df_gemini['model_name'].nunique()}")
    print(f"Models found: {sorted(df_gemini['model_name'].unique())}")

    # Analyze each model separately
    for model_name in sorted(df_gemini['model_name'].unique()):
        if pd.notna(model_name):
            df_model = df_gemini[df_gemini['model_name'] == model_name].copy()
            analyze_model_data(df_model, model_name)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()