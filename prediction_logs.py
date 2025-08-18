import os
import warnings
import argparse
from datetime import datetime, timezone, timedelta

import google.auth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery

warnings.filterwarnings('ignore')

# --- USER INPUT SECTION ---
# BigQuery dataset and table IDs - now using environment variables
dataset_id = os.getenv("DATASET", "MY_DATASET")
project_id = os.getenv("PROJECT_ID")
# --- END USER INPUT SECTION ---

if not project_id:
    # Initialize the BigQuery client
    _, project_id = google.auth.default()

client = bigquery.Client(project=project_id)
prediction_table_id = "prediction"

def parse_arguments():
    """Parse command line arguments for start and end timestamps"""
    parser = argparse.ArgumentParser(description='Analyze prediction log data with customizable time range')

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


print(f"Using dataset_id={dataset_id}, prediction_table_id={prediction_table_id}, project_id={project_id}")
print(f"Analyzing prediction data from {start_filter_timestamp} to {end_filter_timestamp} UTC")
print(f"Time range: {(end_dt - start_dt).days} days, {(end_dt - start_dt).seconds // 3600} hours")

# Define the SQL query to get prediction data
prediction_sql = f"""
  SELECT
    timestamp,
    time_taken_total,
    time_taken_retrieval,
    time_taken_llm,
    app_version,
    tokens_used,
    confidence_score,
    response_type,
    run_type
  FROM
    `{project_id}.{dataset_id}.{prediction_table_id}`
  WHERE
    timestamp BETWEEN '{start_filter_timestamp}' AND '{end_filter_timestamp}'
    AND time_taken_total IS NOT NULL
    AND app_version IS NOT NULL
  ORDER BY timestamp
"""

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

def categorize_latency(latency):
    """Categorize latency into performance buckets"""
    if latency < 3.0:
        return 'Fast (< 3s)'
    elif latency < 5.0:
        return 'Medium (3-5s)'
    elif latency < 10.0:
        return 'Slow (5-10s)'
    elif latency < 20.0:
        return 'Very Slow (10-20s)'
    else:
        return 'Outlier (20s+)'

def analyze_app_version_data(df_app, app_version):
    """Generate comprehensive analysis for a specific app version"""

    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR APP VERSION: {app_version}")
    print(f"{'='*60}")

    if df_app.empty:
        print("No data found for this app version.")
        return

    # Clean data - remove any negative or extremely large values
    df_app = df_app[
        (df_app['time_taken_total'] >= 0) &
        (df_app['time_taken_total'] <= 300) &  # Cap at 5 minutes
        (df_app['time_taken_retrieval'] >= 0) &
        (df_app['time_taken_llm'] >= 0) &
        (df_app['tokens_used'] >= 0)
        ].copy()

    if df_app.empty:
        print("No valid data after cleaning.")
        return

    # === CALCULATE STANDARD DEVIATION STATISTICS ===
    mean_total = df_app['time_taken_total'].mean()
    std_total = df_app['time_taken_total'].std()
    count_total = len(df_app)

    # Define outlier thresholds
    std_2_threshold = mean_total + 2 * std_total
    std_3_threshold = mean_total + 3 * std_total

    # Count requests greater than 2 and 3 STD deviations from the mean
    count_gt_2_std = len(df_app[df_app['time_taken_total'] > std_2_threshold])
    count_gt_3_std = len(df_app[df_app['time_taken_total'] > std_3_threshold])

    # Calculate percentages
    percent_gt_2_std = (count_gt_2_std / count_total) * 100 if count_total > 0 else 0
    percent_gt_3_std = (count_gt_3_std / count_total) * 100 if count_total > 0 else 0

    # Add latency categories
    df_app['latency_category'] = df_app['time_taken_total'].apply(categorize_latency)

    # Add time components for analysis
    df_app['hour'] = df_app['timestamp'].dt.hour
    df_app['request_order'] = range(len(df_app))

    # Calculate time breakdown percentages
    df_app['retrieval_percentage'] = (df_app['time_taken_retrieval'] / df_app['time_taken_total']) * 100
    df_app['llm_percentage'] = (df_app['time_taken_llm'] / df_app['time_taken_total']) * 100
    df_app['other_percentage'] = 100 - df_app['retrieval_percentage'] - df_app['llm_percentage']

    # Print basic statistics
    print(f"\n--- Data Summary for {app_version} ---")
    print(f"Total predictions: {len(df_app)}")
    print(f"Date range: {df_app['timestamp'].min()} to {df_app['timestamp'].max()}")

    print(f"\nTiming statistics:")
    print(f"  Total time - Mean: {mean_total:.3f}s, Std Dev: {std_total:.3f}s")
    print(f"  Retrieval time - Mean: {df_app['time_taken_retrieval'].mean():.3f}s")
    print(f"  LLM time - Mean: {df_app['time_taken_llm'].mean():.3f}s")
    print(f"  > 2 STD ({std_2_threshold:.3f}s): {count_gt_2_std} ({percent_gt_2_std:.2f}%)")
    print(f"  > 3 STD ({std_3_threshold:.3f}s): {count_gt_3_std} ({percent_gt_3_std:.2f}%)")

    # Calculate correlations
    total_tokens_corr = np.corrcoef(df_app['tokens_used'], df_app['time_taken_total'])[0, 1] if len(df_app) > 1 else 0

    # Create comprehensive visualizations
    plt.style.use('default')
    fig = plt.figure(figsize=(32, 28))
    fig.suptitle(f'Prediction Analysis - App Version: {app_version}\n{start_filter_timestamp} to {end_filter_timestamp} UTC',
                 fontsize=18, fontweight='bold', y=0.98)

    # === ROW 1: SUMMARY TABLES ===

    # Table 1: Basic Statistics (Enhanced with STD Dev)
    ax1 = plt.subplot(4, 3, 1)
    ax1.axis('tight')
    ax1.axis('off')

    basic_stats = [
        ['Metric', 'Value'],
        ['Total Predictions', f'{len(df_app):,}'],
        ['Date Range', f'{df_app["timestamp"].min().strftime("%Y-%m-%d %H:%M")} to {df_app["timestamp"].max().strftime("%Y-%m-%d %H:%M")}'],
        ['Mean Total Time', f'{mean_total:.3f}s'],
        ['Std Deviation', f'{std_total:.3f}s'],
        ['Median Total Time', f'{df_app["time_taken_total"].median():.3f}s'],
        ['P95 Total Time', f'{df_app["time_taken_total"].quantile(0.95):.3f}s'],
        ['P99 Total Time', f'{df_app["time_taken_total"].quantile(0.99):.3f}s'],
        ['Max Total Time', f'{df_app["time_taken_total"].max():.3f}s'],
        ['> 2 STD', f'{count_gt_2_std} ({percent_gt_2_std:.2f}%)'],
        ['> 3 STD', f'{count_gt_3_std} ({percent_gt_3_std:.2f}%)'],
    ]

    table1 = ax1.table(cellText=basic_stats[1:], colLabels=basic_stats[0],
                       cellLoc='left', loc='center', colWidths=[0.3, 0.7])
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

    # Table 2: Average Time Breakdown
    ax2 = plt.subplot(4, 3, 2)
    ax2.axis('tight')
    ax2.axis('off')

    avg_retrieval = df_app['time_taken_retrieval'].mean()
    avg_llm = df_app['time_taken_llm'].mean()
    avg_other = mean_total - avg_retrieval - avg_llm

    time_breakdown_stats = [
        ['Component', 'Avg Time', 'Percentage'],
        ['Retrieval', f'{avg_retrieval:.3f}s', f'{df_app["retrieval_percentage"].mean():.1f}%'],
        ['LLM Processing', f'{avg_llm:.3f}s', f'{df_app["llm_percentage"].mean():.1f}%'],
        ['Other/Overhead', f'{avg_other:.3f}s', f'{df_app["other_percentage"].mean():.1f}%'],
        ['Total', f'{mean_total:.3f}s', '100.0%'],
    ]

    table2 = ax2.table(cellText=time_breakdown_stats[1:], colLabels=time_breakdown_stats[0],
                       cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2.0)

    for i in range(len(time_breakdown_stats[0])):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Average Time Breakdown', fontsize=16, fontweight='bold', pad=20)

    # Table 3: Time Breakdown by Category
    ax3 = plt.subplot(4, 3, 3)
    ax3.axis('tight')
    ax3.axis('off')

    # Calculate breakdown by latency category
    category_breakdown = []
    category_order = ['Fast (< 3s)', 'Medium (3-5s)', 'Slow (5-10s)', 'Very Slow (10-20s)', 'Outlier (20s+)']

    breakdown_data = [['Category', 'Count', 'Avg Retrieval', 'Avg LLM', 'Avg Total']]

    for category in category_order:
        cat_data = df_app[df_app['latency_category'] == category]
        if len(cat_data) > 0:
            breakdown_data.append([
                category,
                f'{len(cat_data)}',
                f'{cat_data["time_taken_retrieval"].mean():.2f}s',
                f'{cat_data["time_taken_llm"].mean():.2f}s',
                f'{cat_data["time_taken_total"].mean():.2f}s'
            ])

    table3 = ax3.table(cellText=breakdown_data[1:], colLabels=breakdown_data[0],
                       cellLoc='center', loc='center', colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2.0)

    for i in range(len(breakdown_data[0])):
        table3[(0, i)].set_facecolor('#FF9800')
        table3[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Time Breakdown by Category', fontsize=16, fontweight='bold', pad=20)

    # === ROW 2: DETAILED LATENCY HISTOGRAMS ===

    # Plot 1: Detailed Latency Histogram for time_taken_total (Enhanced with STD Dev)
    ax4 = plt.subplot(4, 3, 4)
    plt.hist(df_app['time_taken_total'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Total Time Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Frequency')

    # Add mean and standard deviation lines
    plt.axvline(mean_total, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_total:.2f}s')
    plt.axvline(mean_total + std_total, color='green', linestyle=':', linewidth=2,
                label=f'1 STD: {std_total:.2f}s')
    plt.axvline(mean_total - std_total, color='green', linestyle=':', linewidth=2)
    plt.axvline(std_2_threshold, color='orange', linestyle='-.', linewidth=2,
                label=f'2 STD: {std_2_threshold:.2f}s')
    plt.axvline(std_3_threshold, color='purple', linestyle='-.', linewidth=2,
                label=f'3 STD: {std_3_threshold:.2f}s')

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 2: Detailed Latency Histogram for time_taken_retrieval
    ax5 = plt.subplot(4, 3, 5)
    plt.hist(df_app['time_taken_retrieval'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Retrieval Time Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Retrieval Time (seconds)')
    plt.ylabel('Frequency')

    mean_retrieval = df_app['time_taken_retrieval'].mean()
    plt.axvline(mean_retrieval, color='red', linestyle='--',
                label=f'Mean: {mean_retrieval:.2f}s')
    plt.axvline(df_app['time_taken_retrieval'].median(), color='green', linestyle='--',
                label=f'Median: {df_app["time_taken_retrieval"].median():.2f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Detailed Latency Histogram for time_taken_llm
    ax6 = plt.subplot(4, 3, 6)
    plt.hist(df_app['time_taken_llm'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('LLM Time Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('LLM Time (seconds)')
    plt.ylabel('Frequency')

    mean_llm = df_app['time_taken_llm'].mean()
    plt.axvline(mean_llm, color='red', linestyle='--',
                label=f'Mean: {mean_llm:.2f}s')
    plt.axvline(df_app['time_taken_llm'].median(), color='green', linestyle='--',
                label=f'Median: {df_app["time_taken_llm"].median():.2f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # === ROW 3: DISTRIBUTION AND CORRELATION ANALYSIS ===

    # Plot 4: Latency Distribution by Category for time_taken_total
    ax7 = plt.subplot(4, 3, 7)
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    category_counts = df_app['latency_category'].value_counts().reindex(category_order, fill_value=0)

    bars = plt.bar(range(len(category_counts)), category_counts.values, color=colors)
    plt.title('Total Time Distribution by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Latency Category')
    plt.ylabel('Count')
    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')

    for bar, count in zip(bars, category_counts.values):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(count), ha='center', va='bottom')

    # Plot 5: Latency vs Token Count
    ax8 = plt.subplot(4, 3, 8)
    scatter = plt.scatter(df_app['tokens_used'], df_app['time_taken_total'],
                          alpha=0.6, s=40, c=df_app['time_taken_total'],
                          cmap='viridis', edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Total Time (seconds)')
    plt.title('Tokens vs Total Time', fontsize=14, fontweight='bold')
    plt.xlabel('Tokens Used')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True, alpha=0.3)

    # Add correlation text
    plt.text(0.05, 0.95, f'Correlation: {total_tokens_corr:.3f}',
             transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Safe trend line
    z, p = safe_polyfit(df_app['tokens_used'], df_app['time_taken_total'])
    if z is not None and p is not None:
        plt.plot(df_app['tokens_used'], p(df_app['tokens_used']),
                 "r--", alpha=0.8, linewidth=2)

    # Plot 6: Request Order vs Latency
    ax9 = plt.subplot(4, 3, 9)
    df_sorted = df_app.sort_values('timestamp').reset_index(drop=True)
    df_sorted['request_number'] = range(1, len(df_sorted) + 1)

    plt.scatter(df_sorted['request_number'], df_sorted['time_taken_total'],
                alpha=0.6, s=15, c=df_sorted['time_taken_total'], cmap='viridis')
    plt.plot(df_sorted['request_number'], df_sorted['time_taken_total'],
             color='red', alpha=0.4, linewidth=0.8)

    window_size = max(10, len(df_sorted) // 30)
    if window_size > 1:
        moving_avg = df_sorted['time_taken_total'].rolling(window=window_size, center=True).mean()
        plt.plot(df_sorted['request_number'], moving_avg,
                 color='blue', linewidth=2, alpha=0.8, label='Moving Average')

    plt.title('Request Order vs Total Time', fontsize=14, fontweight='bold')
    plt.xlabel('Request Number')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True, alpha=0.3)
    if window_size > 1:
        plt.legend()

    # === ROW 4: ADVANCED ANALYSIS ===

    # Plot 7: Token Count Distribution
    ax10 = plt.subplot(4, 3, 10)
    plt.hist(df_app['tokens_used'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Token Usage Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Tokens Used')
    plt.ylabel('Frequency')
    plt.axvline(df_app['tokens_used'].mean(), color='red', linestyle='--',
                label=f'Mean: {df_app["tokens_used"].mean():.1f}')
    plt.axvline(df_app['tokens_used'].median(), color='green', linestyle='--',
                label=f'Median: {df_app["tokens_used"].median():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Latency Distribution by Hour for each category
    ax11 = plt.subplot(4, 3, 11)
    unique_hours = sorted(df_app['hour'].unique())

    hourly_data = []
    hour_labels = []
    for h in unique_hours:
        data = df_app[df_app['hour'] == h]['time_taken_total'].values
        if len(data) > 0:
            hourly_data.append(data)
            hour_labels.append(f"{h:02d}:00")

    if len(hourly_data) > 0:
        plt.boxplot(hourly_data, labels=hour_labels)
        plt.title(f'Total Time Distribution by Hour\n({len(unique_hours)} hours with data)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No hourly data available', ha='center', va='center', transform=ax11.transAxes)
        plt.title('Total Time Distribution by Hour', fontsize=14, fontweight='bold')

    # Plot 9: Time Breakdown Pie Chart
    ax12 = plt.subplot(4, 3, 12)
    avg_retrieval = df_app['time_taken_retrieval'].mean()
    avg_llm = df_app['time_taken_llm'].mean()
    avg_other = mean_total - avg_retrieval - avg_llm

    sizes = [avg_retrieval, avg_llm, max(0, avg_other)]
    labels = ['Retrieval', 'LLM', 'Other']
    colors = ['lightcoral', 'lightblue', 'lightgreen']

    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Average Time Breakdown', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No valid time breakdown data', ha='center', va='center', transform=ax12.transAxes)
        plt.title('Average Time Breakdown', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=4.0)

    # Save plot to file
    safe_app_version = app_version.replace(".", "_").replace("-", "_").replace("/", "_")
    filename = os.path.join(plots_dir,  f'prediction_analysis_{safe_app_version}_{timestamp_str}')

    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')

    # plt.show()  # Commented out to match the Gemini analysis behavior

    # Print enhanced insights
    print(f"\n--- Key Insights for {app_version} ---")

    print(f"1. Performance Overview:")
    print(f"   - Average total time: {mean_total:.3f}s")
    print(f"   - Standard deviation: {std_total:.3f}s")
    print(f"   - 95th percentile: {df_app['time_taken_total'].quantile(0.95):.3f}s")
    print(f"   - Requests > 2 STD: {count_gt_2_std} ({percent_gt_2_std:.2f}%)")
    print(f"   - Requests > 3 STD: {count_gt_3_std} ({percent_gt_3_std:.2f}%)")

    if percent_gt_2_std > 5:
        print(f"   ⚠️  WARNING: High percentage of requests beyond 2 STD - investigate outliers")
    if percent_gt_3_std > 1:
        print(f"   ⚠️  WARNING: Significant percentage of requests beyond 3 STD - potential system issues")

    print(f"2. Time Breakdown:")
    print(f"   - Retrieval takes {df_app['retrieval_percentage'].mean():.1f}% of total time")
    print(f"   - LLM processing takes {df_app['llm_percentage'].mean():.1f}% of total time")
    print(f"   - Other overhead takes {df_app['other_percentage'].mean():.1f}% of total time")

    print(f"3. Token Analysis:")
    print(f"   - Average tokens per prediction: {df_app['tokens_used'].mean():.1f}")
    print(f"   - Token-Total time correlation: {total_tokens_corr:.3f}")

    if abs(total_tokens_corr) > 0.3:
        direction = "positive" if total_tokens_corr > 0 else "negative"
        strength = "strong" if abs(total_tokens_corr) > 0.7 else "moderate"
        print(f"   - {strength.capitalize()} {direction} correlation between tokens and total time")

    print(f"Plots saved as:")
    print(f" - {filename}.png")
    print(f" - {filename}.pdf")

try:
    # Run the query and get the result as a DataFrame
    print("Executing BigQuery...")
    df_prediction = client.query(prediction_sql).to_dataframe()

    # Drop rows where required fields are missing
    df_prediction.dropna(subset=['time_taken_total', 'timestamp', 'app_version'], inplace=True)

    if df_prediction.empty:
        print("No data found for the specified time range.")
        exit()

    # Convert timestamp to datetime for analysis
    df_prediction['timestamp'] = pd.to_datetime(df_prediction['timestamp'])

    # Print overall summary
    print(f"\n--- OVERALL SUMMARY ---")
    print(f"Total predictions: {len(df_prediction)}")
    print(f"Date range: {df_prediction['timestamp'].min()} to {df_prediction['timestamp'].max()}")
    print(f"Unique app versions: {df_prediction['app_version'].nunique()}")
    print(f"App versions found: {sorted(df_prediction['app_version'].unique())}")

    # Show distribution by app version
    print(f"\nPredictions per app version:")
    version_counts = df_prediction['app_version'].value_counts()
    for version, count in version_counts.items():
        print(f"  {version}: {count:,} predictions ({count/len(df_prediction)*100:.1f}%)")

    # Analyze each app version separately
    for app_version in sorted(df_prediction['app_version'].unique()):
        if pd.notna(app_version):
            df_app = df_prediction[df_prediction['app_version'] == app_version].copy()
            analyze_app_version_data(df_app, app_version)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()