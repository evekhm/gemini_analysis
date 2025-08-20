import json
import warnings
from datetime import datetime, timezone, timedelta
import argparse

import google.auth
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
import os

warnings.filterwarnings('ignore')

# Add this import for statistical tests
try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not available, statistical tests will be skipped")
    stats = None

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

def get_project_name(project_id):
    """Try to get project name from project ID using Google Cloud Resource Manager API"""
    try:
        from google.cloud import resourcemanager_v3
        client = resourcemanager_v3.ProjectsClient()
        project_name = f"projects/{project_id}"
        project = client.get_project(name=project_name)
        return project.display_name if project.display_name else project_id
    except Exception as e:
        print(f"Could not retrieve project name: {e}")
        return project_id

def add_page_header(fig, project_id, start_time, end_time, generation_time, title_suffix=""):
    """Add project information and metadata to the top of each page"""
    try:
        project_name = get_project_name(project_id)
        if project_name != project_id:
            project_info = f"Project: {project_name} (ID: {project_id})"
        else:
            project_info = f"Project ID: {project_id}"
    except:
        project_info = f"Project ID: {project_id}"

    header_text = f"""{project_info}
Analysis Period: {start_time} to {end_time} UTC
Generated: {generation_time} UTC{title_suffix}"""

    fig.text(0.02, 0.98, header_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

def create_latency_token_plots(df_tokens, model_name, start_time, end_time, generation_time):
    """Create standalone high-resolution latency vs token plots with multiple scales"""
    if df_tokens is None or len(df_tokens) == 0:
        print(f"No token data available for {model_name}")
        return

    # Filter positive tokens for log scale
    df_tokens_positive = df_tokens[df_tokens['output_tokens'] > 0].copy()
    if len(df_tokens_positive) == 0:
        print(f"No positive token data available for {model_name}")
        return

    # Create color mapping based on input tokens
    input_tokens_available = df_tokens_positive['input_tokens'].notna().any()
    if input_tokens_available:
        color_data = df_tokens_positive['input_tokens']
        color_label = 'Input Tokens'
        color_map = 'plasma'  # Good color map for input tokens
    else:
        color_data = df_tokens_positive['latency_seconds']
        color_label = 'Latency (seconds)'
        color_map = 'viridis'

    # Calculate correlation
    token_latency_corr = np.corrcoef(df_tokens_positive['output_tokens'],
                                     df_tokens_positive['latency_seconds'])[0, 1]

    safe_model_name = model_name.replace("-", "_").replace(".", "_")

    # Create figure with 2x2 subplots for different scale combinations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Add page header
    add_page_header(fig, project_id, start_time, end_time, generation_time,
                    f" - Latency vs Output Tokens Analysis")

    fig.suptitle(f'Latency vs Output Token Count Analysis - {model_name}\n'
                 f'Correlation: {token_latency_corr:.3f} | Total Requests: {len(df_tokens_positive):,}',
                 fontsize=16, fontweight='bold', y=0.92)

    # Plot configurations: (x_scale, y_scale, title_suffix)
    plot_configs = [
        ('linear', 'linear', 'Linear-Linear'),
        ('log', 'linear', 'Log-Linear'),
        ('linear', 'log', 'Linear-Log'),
        ('log', 'log', 'Log-Log')
    ]

    for idx, (x_scale, y_scale, title_suffix) in enumerate(plot_configs):
        ax = axes[idx // 2, idx % 2]

        # Create scatter plot with smaller dots
        scatter = ax.scatter(df_tokens_positive['output_tokens'],
                             df_tokens_positive['latency_seconds'],
                             c=color_data, cmap=color_map, alpha=0.6, s=8,  # Smaller dots (s=8)
                             edgecolors='black', linewidth=0.1)

        # Set scales
        if x_scale == 'log':
            ax.set_xscale('log')
        if y_scale == 'log':
            ax.set_yscale('log')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, fontsize=10)

        # Labels and title
        ax.set_xlabel(f'Output Token Count ({x_scale.title()} Scale)', fontsize=12)
        ax.set_ylabel(f'Latency ({y_scale.title()} Scale)', fontsize=12)
        ax.set_title(f'{title_suffix} Scale\nCorr: {token_latency_corr:.3f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line (handle different scales appropriately)
        try:
            if x_scale == 'log' and y_scale == 'log':
                # Log-log: use log of both variables
                log_x = np.log10(df_tokens_positive['output_tokens'])
                log_y = np.log10(df_tokens_positive['latency_seconds'])
                z, p = safe_polyfit(log_x, log_y)
                if z is not None:
                    x_trend = np.logspace(np.log10(df_tokens_positive['output_tokens'].min()),
                                          np.log10(df_tokens_positive['output_tokens'].max()), 100)
                    y_trend = 10**(z[0] * np.log10(x_trend) + z[1])
                    ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2)
            elif x_scale == 'log':
                # Log-linear: use log of x
                log_x = np.log10(df_tokens_positive['output_tokens'])
                z, p = safe_polyfit(log_x, df_tokens_positive['latency_seconds'])
                if z is not None:
                    x_trend = np.logspace(np.log10(df_tokens_positive['output_tokens'].min()),
                                          np.log10(df_tokens_positive['output_tokens'].max()), 100)
                    y_trend = z[0] * np.log10(x_trend) + z[1]
                    ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2)
            elif y_scale == 'log':
                # Linear-log: use log of y
                log_y = np.log10(df_tokens_positive['latency_seconds'])
                z, p = safe_polyfit(df_tokens_positive['output_tokens'], log_y)
                if z is not None:
                    x_trend = np.linspace(df_tokens_positive['output_tokens'].min(),
                                          df_tokens_positive['output_tokens'].max(), 100)
                    y_trend = 10**(z[0] * x_trend + z[1])
                    ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2)
            else:
                # Linear-linear: standard approach
                z, p = safe_polyfit(df_tokens_positive['output_tokens'],
                                    df_tokens_positive['latency_seconds'])
                if z is not None and p is not None:
                    ax.plot(df_tokens_positive['output_tokens'],
                            p(df_tokens_positive['output_tokens']),
                            "r--", alpha=0.8, linewidth=2)
        except Exception as e:
            print(f"Could not add trend line for {title_suffix}: {e}")

        # Add statistics text box
        stats_text = f'N: {len(df_tokens_positive):,}\n'
        if input_tokens_available:
            stats_text += f'Input tokens: {df_tokens_positive["input_tokens"].min():.0f}-{df_tokens_positive["input_tokens"].max():.0f}\n'
        stats_text += f'Output tokens: {df_tokens_positive["output_tokens"].min():.0f}-{df_tokens_positive["output_tokens"].max():.0f}\n'
        stats_text += f'Latency: {df_tokens_positive["latency_seconds"].min():.2f}-{df_tokens_positive["latency_seconds"].max():.2f}s'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Save high-resolution plots
    filename_base = os.path.join(plots_dir, f'latency_vs_tokens_{safe_model_name}_{timestamp_str}')

    # Save as high-resolution PNG (4K equivalent)
    plt.savefig(f'{filename_base}_4K.png', dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename_base}.pdf', bbox_inches='tight', facecolor='white')

    if args.show:
        plt.show()

    print(f"High-resolution latency vs token plots saved:")
    print(f" - {filename_base}_4K.png (4K resolution)")
    print(f" - {filename_base}.png (standard resolution)")
    print(f" - {filename_base}.pdf")


def create_hourly_analysis_with_weekday(df_model, model_name, start_time, end_time, generation_time):
    """Create enhanced hourly analysis differentiated by working vs non-working days"""
    if df_model.empty:
        print(f"No data available for {model_name}")
        return

    # Add day of week information
    df_model = df_model.copy()
    df_model['hour'] = df_model['logging_time'].dt.hour
    df_model['day_of_week'] = df_model['logging_time'].dt.day_name()
    df_model['is_weekend'] = df_model['logging_time'].dt.weekday >= 5  # Saturday=5, Sunday=6
    df_model['day_type'] = df_model['is_weekend'].map({True: 'Non-Working Days', False: 'Working Days'})

    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Add page header
    add_page_header(fig, project_id, start_time, end_time, generation_time,
                    f" - Hourly Analysis by Day Type")

    fig.suptitle(f'Latency Distribution by Hour and Day Type - {model_name}',
                 fontsize=16, fontweight='bold', y=0.95)

    # Separate data by working vs non-working days
    working_data = df_model[df_model['day_type'] == 'Working Days']
    non_working_data = df_model[df_model['day_type'] == 'Non-Working Days']

    # Plot 1: Request Count by Hour - Working Days
    ax1 = axes[0, 0]
    if len(working_data) > 0:
        working_hourly = working_data['hour'].value_counts().reindex(range(24), fill_value=0)
        bars1 = ax1.bar(working_hourly.index, working_hourly.values, alpha=0.7, color='green', edgecolor='black')
        ax1.set_title('Request Count by Hour\n(Working Days)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Request Count')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))

        # Add count labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(working_hourly.values) * 0.01,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No working day data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Request Count by Hour\n(Working Days)', fontsize=14, fontweight='bold')

    # Plot 2: Request Count by Hour - Non-Working Days
    ax2 = axes[0, 1]
    if len(non_working_data) > 0:
        non_working_hourly = non_working_data['hour'].value_counts().reindex(range(24), fill_value=0)
        bars2 = ax2.bar(non_working_hourly.index, non_working_hourly.values, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('Request Count by Hour\n(Non-Working Days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Request Count')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))

        # Add count labels on bars
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(non_working_hourly.values) * 0.01,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No non-working day data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Request Count by Hour\n(Non-Working Days)', fontsize=14, fontweight='bold')

    # Plot 3: Mean Latency by Hour - Working Days
    ax3 = axes[0, 2]
    if len(working_data) > 0:
        working_latency = working_data.groupby('hour')['latency_seconds'].agg(['mean', 'count']).reset_index()
        working_latency_filtered = working_latency[working_latency['count'] >= 2]

        if len(working_latency_filtered) > 0:
            bars3 = ax3.bar(working_latency_filtered['hour'], working_latency_filtered['mean'],
                            alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('Mean Latency by Hour\n(Working Days)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Mean Latency (seconds)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, 24, 2))

            # Add latency values on bars
            for bar, latency in zip(bars3, working_latency_filtered['mean']):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(working_latency_filtered['mean']) * 0.01,
                         f'{latency:.2f}s', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for hourly latency analysis',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Mean Latency by Hour\n(Working Days)', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No working day data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Mean Latency by Hour\n(Working Days)', fontsize=14, fontweight='bold')

    # Plot 4: Mean Latency by Hour - Non-Working Days
    ax4 = axes[1, 0]
    if len(non_working_data) > 0:
        non_working_latency = non_working_data.groupby('hour')['latency_seconds'].agg(['mean', 'count']).reset_index()
        non_working_latency_filtered = non_working_latency[non_working_latency['count'] >= 2]

        if len(non_working_latency_filtered) > 0:
            bars4 = ax4.bar(non_working_latency_filtered['hour'], non_working_latency_filtered['mean'],
                            alpha=0.7, color='lightsalmon', edgecolor='black')
            ax4.set_title('Mean Latency by Hour\n(Non-Working Days)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Mean Latency (seconds)')
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(range(0, 24, 2))

            # Add latency values on bars
            for bar, latency in zip(bars4, non_working_latency_filtered['mean']):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(non_working_latency_filtered['mean']) * 0.01,
                         f'{latency:.2f}s', ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for hourly latency analysis',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Mean Latency by Hour\n(Non-Working Days)', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No non-working day data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Mean Latency by Hour\n(Non-Working Days)', fontsize=14, fontweight='bold')

    # Plot 5: Box plots by day of week
    ax5 = axes[1, 1]
    days_with_data = df_model['day_of_week'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_to_plot = [day for day in day_order if day in days_with_data.index and days_with_data[day] >= 3]

    if len(days_to_plot) > 0:
        day_data = []
        day_labels = []
        for day in days_to_plot:
            data = df_model[df_model['day_of_week'] == day]['latency_seconds'].values
            if len(data) >= 3:
                day_data.append(data)
                day_labels.append(f"{day}\n(n={len(data)})")

        if len(day_data) > 0:
            box_plot = ax5.boxplot(day_data, labels=day_labels, patch_artist=True)

            # Color weekends differently
            for i, (patch, day) in enumerate(zip(box_plot['boxes'], days_to_plot)):
                if day in ['Saturday', 'Sunday']:
                    patch.set_facecolor('orange')
                    patch.set_alpha(0.7)
                else:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)

            ax5.set_title('Latency Distribution by Day of Week', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Latency (seconds)')
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for daily box plots',
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Latency Distribution by Day of Week', fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for daily analysis',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Latency Distribution by Day of Week', fontsize=14, fontweight='bold')

    # Plot 6: Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')

    # Calculate summary statistics
    total_requests = len(df_model)
    working_requests = len(working_data)
    non_working_requests = len(non_working_data)

    working_mean_latency = working_data['latency_seconds'].mean() if len(working_data) > 0 else 0
    non_working_mean_latency = non_working_data['latency_seconds'].mean() if len(non_working_data) > 0 else 0

    # Most active hours
    if len(working_data) > 0:
        working_hourly_counts = working_data['hour'].value_counts()
        most_active_working_hour = working_hourly_counts.idxmax()
        most_active_working_count = working_hourly_counts.max()
    else:
        most_active_working_hour = "N/A"
        most_active_working_count = 0

    if len(non_working_data) > 0:
        non_working_hourly_counts = non_working_data['hour'].value_counts()
        most_active_non_working_hour = non_working_hourly_counts.idxmax()
        most_active_non_working_count = non_working_hourly_counts.max()
    else:
        most_active_non_working_hour = "N/A"
        most_active_non_working_count = 0

    summary_stats = [
        ['Metric', 'Value'],
        ['Total Requests', f'{total_requests:,}'],
        ['Working Day Requests', f'{working_requests:,} ({working_requests/total_requests*100:.1f}%)'],
        ['Non-Working Day Requests', f'{non_working_requests:,} ({non_working_requests/total_requests*100:.1f}%)'],
        ['Most Active Hour (Working)', f'{most_active_working_hour}:00 ({most_active_working_count} requests)'],
        ['Most Active Hour (Non-Working)', f'{most_active_non_working_hour}:00 ({most_active_non_working_count} requests)'],
        ['Working Days Mean Latency', f'{working_mean_latency:.3f}s'],
        ['Non-Working Days Mean Latency', f'{non_working_mean_latency:.3f}s'],
        ['Latency Difference', f'{non_working_mean_latency - working_mean_latency:+.3f}s'],
    ]

    table = ax6.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                      cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Style the header
    for i in range(len(summary_stats[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the plot
    safe_model_name = model_name.replace("-", "_").replace(".", "_")
    filename = os.path.join(plots_dir, f'hourly_weekday_analysis_{safe_model_name}_{timestamp_str}')

    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')

    if args.show:
        plt.show()

    print(f"Hourly weekday analysis saved:")
    print(f" - {filename}.png")
    print(f" - {filename}.pdf")


def parse_arguments():
    """Parse command line arguments for start and end timestamps"""
    parser = argparse.ArgumentParser(description='Analyze Gemini log data with customizable time range and bucket sizes')

    # Default start time: 90 days ago
    default_start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    # Default end time: current UTC time
    default_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    parser.add_argument('--show',
                        default=False, type=bool,
                        help=f'Show plots (default: {False})')

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

    # NEW: Bucket configuration arguments
    parser.add_argument('--bucket-sizes', '-b',
                        type=str,
                        default="300,600",
                        help='Comma-separated list of bucket sizes in seconds (default: "300,600" for 5min,10min). Examples: "60,300" for 1min,5min or "10,60,300" for 10s,1min,5min')

    parser.add_argument('--bucket-method', '-m',
                        choices=['start_time', 'overlap'],
                        default='overlap',
                        help='Method for bucket assignment: "start_time" (faster, use request start time only) or "overlap" (slower, check for overlapping execution time). Default: overlap')

    return parser.parse_args()

def validate_timestamp(timestamp_str):
    """Validate timestamp format"""
    try:
        datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False

def parse_bucket_sizes(bucket_sizes_str):
    """Parse bucket sizes from command line argument"""
    try:
        bucket_sizes = [int(x.strip()) for x in bucket_sizes_str.split(',')]
        # Validate bucket sizes
        for size in bucket_sizes:
            if size <= 0:
                raise ValueError(f"Bucket size must be positive, got: {size}")
        return bucket_sizes
    except ValueError as e:
        print(f"Error parsing bucket sizes: {e}")
        print("Please provide comma-separated positive integers (e.g., '60,300,600')")
        exit(1)

# Parse command line arguments
args = parse_arguments()

# Parse bucket sizes
bucket_sizes = parse_bucket_sizes(args.bucket_sizes)
bucket_method = args.bucket_method

print(f"Using bucket sizes: {bucket_sizes} seconds ({[f'{b//60}min' if b >= 60 else f'{b}s' for b in bucket_sizes]})")
print(f"Using bucket method: {bucket_method}")

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

def extract_token_counts(full_response_str):
    """Extract input, output, and total token counts from full_response JSON string"""
    try:
        if pd.isna(full_response_str) or full_response_str == '':
            return None, None, None

        # Parse JSON
        response_json = json.loads(full_response_str)

        # Extract token counts
        usage_metadata = response_json.get('usageMetadata', {})
        input_tokens = usage_metadata.get('promptTokenCount')  # Input tokens
        output_tokens = usage_metadata.get('candidatesTokenCount')  # Output tokens
        total_tokens = usage_metadata.get('totalTokenCount')  # Total tokens

        return input_tokens, output_tokens, total_tokens
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, None, None

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

def analyze_concurrency_hypothesis(df_model, model_name):
    """
    Focused analysis to test the hypothesis that higher concurrency leads to longer response times.
    Uses multiple approaches to reduce noise and highlight the relationship.
    """

    print(f"\n{'='*60}")
    print(f"CONCURRENCY HYPOTHESIS TESTING FOR MODEL: {model_name}")
    print(f"{'='*60}")

    if df_model.empty:
        print("No data found for this model.")
        return None

    # Calculate request start and end times
    df_model = df_model.copy()
    df_model['request_start'] = df_model['logging_time'] - pd.to_timedelta(df_model['latency_seconds'], unit='s')
    df_model['request_end'] = df_model['logging_time']

    # Sort by start time for analysis
    df_sorted = df_model.sort_values('request_start').reset_index(drop=True)

    # Method 1: Calculate exact concurrency for each request
    print("Calculating exact concurrency for each request...")
    concurrency_data = []

    for idx, row in df_sorted.iterrows():
        # Count how many other requests were executing when this request started
        concurrent_count = len(df_sorted[
                                   (df_sorted['request_start'] <= row['request_start']) &
                                   (df_sorted['request_end'] > row['request_start']) &
                                   (df_sorted.index != idx)  # Exclude the current request
                                   ]) + 1  # +1 to include the current request

        concurrency_data.append({
            'request_idx': idx,
            'request_start': row['request_start'],
            'latency_seconds': row['latency_seconds'],
            'concurrent_count': concurrent_count,
            'output_tokens': row.get('output_tokens', 0)
        })

    df_concurrency = pd.DataFrame(concurrency_data)

    # Method 2: Focus on high vs low concurrency periods
    concurrency_threshold = df_concurrency['concurrent_count'].quantile(0.75)  # Top 25%
    high_concurrency = df_concurrency[df_concurrency['concurrent_count'] >= concurrency_threshold]
    low_concurrency = df_concurrency[df_concurrency['concurrent_count'] < concurrency_threshold]

    # Method 3: Moving window analysis (5-request windows)
    window_size = min(5, len(df_sorted) // 10)  # Adaptive window size
    if window_size >= 3:
        df_sorted['concurrency_window'] = df_sorted['latency_seconds'].rolling(window=window_size, center=True).mean()
        df_sorted['request_rate'] = 1.0  # Will calculate actual rate

        # Calculate request rate in each window
        for i in range(window_size, len(df_sorted) - window_size):
            window_start = df_sorted.iloc[i - window_size//2]['request_start']
            window_end = df_sorted.iloc[i + window_size//2]['request_start']
            time_span = (window_end - window_start).total_seconds()
            if time_span > 0:
                df_sorted.loc[i, 'request_rate'] = window_size / time_span

    # Statistical Analysis
    correlation_exact = np.corrcoef(df_concurrency['concurrent_count'], df_concurrency['latency_seconds'])[0, 1]

    print(f"\n--- Statistical Analysis ---")
    print(f"Total requests analyzed: {len(df_concurrency)}")
    print(f"Concurrency range: {df_concurrency['concurrent_count'].min()} to {df_concurrency['concurrent_count'].max()}")
    print(f"Exact concurrency vs latency correlation: {correlation_exact:.3f}")

    performance_impact = 0
    if len(high_concurrency) > 0 and len(low_concurrency) > 0:
        high_conc_mean_latency = high_concurrency['latency_seconds'].mean()
        low_conc_mean_latency = low_concurrency['latency_seconds'].mean()
        performance_impact = ((high_conc_mean_latency - low_conc_mean_latency) / low_conc_mean_latency) * 100

        print(f"\nHigh vs Low Concurrency Comparison:")
        print(f"  High concurrency (≥{concurrency_threshold:.0f}): {len(high_concurrency)} requests, {high_conc_mean_latency:.3f}s avg latency")
        print(f"  Low concurrency (<{concurrency_threshold:.0f}): {len(low_concurrency)} requests, {low_conc_mean_latency:.3f}s avg latency")
        print(f"  Performance impact: {performance_impact:+.1f}%")

        # Statistical significance test (t-test)
        if stats is not None:
            t_stat, p_value = stats.ttest_ind(high_concurrency['latency_seconds'], low_concurrency['latency_seconds'])
            print(f"  T-test p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

    # Create focused visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Concurrency Hypothesis Testing - {model_name}', fontsize=16, fontweight='bold')

    # Plot 1: Exact concurrency vs latency (scatter with trend)
    axes[0, 0].scatter(df_concurrency['concurrent_count'], df_concurrency['latency_seconds'],
                       alpha=0.6, s=30, c=df_concurrency['latency_seconds'], cmap='viridis')
    axes[0, 0].set_title(f'Exact Concurrency vs Latency\nCorr: {correlation_exact:.3f}')
    axes[0, 0].set_xlabel('Concurrent Requests')
    axes[0, 0].set_ylabel('Latency (seconds)')
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    z, p = safe_polyfit(df_concurrency['concurrent_count'], df_concurrency['latency_seconds'])
    if z is not None and p is not None:
        axes[0, 0].plot(df_concurrency['concurrent_count'], p(df_concurrency['concurrent_count']),
                        "r--", alpha=0.8, linewidth=2)

    # Plot 2: Box plot of latency by concurrency level with automatic bucketing
    concurrency_levels = sorted(df_concurrency['concurrent_count'].unique())

    def create_concurrency_buckets(df_concurrency, concurrency_levels, max_buckets=8):
        """Automatically create feasible concurrency buckets for box plot"""

        if len(concurrency_levels) <= max_buckets:
            # If we have few enough levels, use them directly
            latency_by_concurrency = []
            level_labels = []

            for level in concurrency_levels:
                level_data = df_concurrency[df_concurrency['concurrent_count'] == level]['latency_seconds'].values
                if len(level_data) >= 3:  # Need minimum data for box plot
                    latency_by_concurrency.append(level_data)
                    level_labels.append(str(level))

            return latency_by_concurrency, level_labels, "individual levels"

        else:
            # Too many levels - create buckets
            min_conc = df_concurrency['concurrent_count'].min()
            max_conc = df_concurrency['concurrent_count'].max()

            # Try different bucketing strategies
            strategies = [
                # Strategy 1: Equal-width buckets
                {
                    'name': 'equal-width',
                    'bins': np.linspace(min_conc, max_conc + 1, max_buckets + 1)
                },
                # Strategy 2: Quantile-based buckets
                {
                    'name': 'quantile',
                    'bins': df_concurrency['concurrent_count'].quantile(np.linspace(0, 1, max_buckets + 1)).values
                },
                # Strategy 3: Custom buckets based on data distribution
                {
                    'name': 'adaptive',
                    'bins': None  # Will be calculated below
                }
            ]

            # For adaptive strategy, create buckets that ensure reasonable sample sizes
            if len(concurrency_levels) > max_buckets:
                # Group levels to ensure each bucket has reasonable sample size
                min_samples_per_bucket = max(10, len(df_concurrency) // (max_buckets * 2))

                adaptive_bins = [min_conc]
                current_count = 0

                for level in concurrency_levels:
                    level_count = len(df_concurrency[df_concurrency['concurrent_count'] == level])
                    current_count += level_count

                    if current_count >= min_samples_per_bucket and len(adaptive_bins) < max_buckets:
                        adaptive_bins.append(level + 0.5)  # Add 0.5 to include the current level
                        current_count = 0

                adaptive_bins.append(max_conc + 1)
                strategies[2]['bins'] = np.array(adaptive_bins)

            # Try each strategy and pick the best one
            best_strategy = None
            best_result = None

            for strategy in strategies:
                if strategy['bins'] is None:
                    continue

                try:
                    # Create buckets using pd.cut
                    df_temp = df_concurrency.copy()
                    df_temp['concurrency_bucket'] = pd.cut(df_temp['concurrent_count'],
                                                           bins=strategy['bins'],
                                                           include_lowest=True,
                                                           duplicates='drop')

                    # Extract data for each bucket
                    latency_by_bucket = []
                    bucket_labels = []

                    for bucket in df_temp['concurrency_bucket'].cat.categories:
                        bucket_data = df_temp[df_temp['concurrency_bucket'] == bucket]['latency_seconds'].values
                        if len(bucket_data) >= 3:  # Minimum data for meaningful box plot
                            latency_by_bucket.append(bucket_data)
                            # Create readable label
                            left = int(bucket.left)
                            right = int(bucket.right)
                            if left == right - 1:
                                bucket_labels.append(str(left))
                            else:
                                bucket_labels.append(f'{left}-{right-1}')

                    # Check if this strategy produces good results
                    if len(latency_by_bucket) >= 2:  # Need at least 2 buckets for comparison
                        best_strategy = strategy['name']
                        best_result = (latency_by_bucket, bucket_labels)
                        break  # Use the first working strategy

                except Exception as e:
                    print(f"Warning: Bucketing strategy '{strategy['name']}' failed: {e}")
                    continue

            if best_result:
                return best_result[0], best_result[1], f"auto-bucketed ({best_strategy})"
            else:
                return [], [], "bucketing failed"

    # Apply automatic bucketing
    latency_by_concurrency, level_labels, bucket_method = create_concurrency_buckets(df_concurrency, concurrency_levels)

    if len(latency_by_concurrency) > 1:
        print(f"Created box plot using {bucket_method} with {len(level_labels)} buckets")

        box_plot = axes[0, 1].boxplot(latency_by_concurrency, labels=level_labels)
        axes[0, 1].set_title(f'Latency Distribution by Concurrency Level\n({bucket_method})')
        axes[0, 1].set_xlabel('Concurrent Requests')
        axes[0, 1].set_ylabel('Latency (seconds)')
        axes[0, 1].grid(True, alpha=0.3)

        # Rotate labels if there are many
        if len(level_labels) > 6:
            axes[0, 1].tick_params(axis='x', rotation=45)

    elif len(latency_by_concurrency) == 1:
        axes[0, 1].text(0.5, 0.5, f'Only one concurrency bucket with sufficient data\n({bucket_method})',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Latency by Concurrency Level')
    else:
        axes[0, 1].text(0.5, 0.5, f'Insufficient data for box plot\n({bucket_method})',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Latency by Concurrency Level')

    # Plot 3: High vs Low concurrency comparison
    if len(high_concurrency) > 0 and len(low_concurrency) > 0:
        axes[0, 2].hist([low_concurrency['latency_seconds'], high_concurrency['latency_seconds']],
                        bins=20, alpha=0.7, label=['Low Concurrency', 'High Concurrency'],
                        color=['green', 'red'])
        axes[0, 2].axvline(low_conc_mean_latency, color='green', linestyle='--', linewidth=2, alpha=0.8)
        axes[0, 2].axvline(high_conc_mean_latency, color='red', linestyle='--', linewidth=2, alpha=0.8)
        axes[0, 2].set_title(f'High vs Low Concurrency\nImpact: {performance_impact:+.1f}%')
        axes[0, 2].set_xlabel('Latency (seconds)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('High vs Low Concurrency')

    # Plot 4: Timeline showing concurrency and latency together
    axes[1, 0].plot(df_concurrency['request_start'], df_concurrency['concurrent_count'],
                    label='Concurrency', alpha=0.7, color='blue')
    ax_twin = axes[1, 0].twinx()
    ax_twin.plot(df_concurrency['request_start'], df_concurrency['latency_seconds'],
                 label='Latency', alpha=0.7, color='red', linewidth=1)
    axes[1, 0].set_title('Concurrency and Latency Over Time')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Concurrent Requests', color='blue')
    ax_twin.set_ylabel('Latency (seconds)', color='red')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Moving average analysis (if we have enough data)
    if window_size >= 3 and 'concurrency_window' in df_sorted.columns:
        valid_data = df_sorted.dropna(subset=['concurrency_window'])
        if len(valid_data) > 10:
            axes[1, 1].scatter(valid_data['request_rate'], valid_data['concurrency_window'],
                               alpha=0.6, s=30, c=valid_data['concurrency_window'], cmap='viridis')
            axes[1, 1].set_title(f'Request Rate vs Avg Latency\n({window_size}-request windows)')
            axes[1, 1].set_xlabel('Request Rate (req/sec)')
            axes[1, 1].set_ylabel('Average Latency (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for moving average', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Request Rate vs Latency')
    else:
        axes[1, 1].text(0.5, 0.5, 'Window size too small', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Request Rate vs Latency')

    # Plot 6: Latency percentiles by concurrency level (using the same bucketing as box plot)
    if len(latency_by_concurrency) > 1:
        percentile_data = []
        for i, (bucket_data, label) in enumerate(zip(latency_by_concurrency, level_labels)):
            if len(bucket_data) >= 5:  # Need minimum data for percentiles
                percentile_data.append({
                    'concurrency_label': label,
                    'concurrency_numeric': i,  # Use index for plotting
                    'p50': np.percentile(bucket_data, 50),
                    'p90': np.percentile(bucket_data, 90),
                    'p95': np.percentile(bucket_data, 95),
                    'count': len(bucket_data)
                })

        if len(percentile_data) > 1:
            df_percentiles = pd.DataFrame(percentile_data)
            axes[1, 2].plot(df_percentiles['concurrency_numeric'], df_percentiles['p50'], 'o-', label='P50', linewidth=2)
            axes[1, 2].plot(df_percentiles['concurrency_numeric'], df_percentiles['p90'], 's-', label='P90', linewidth=2)
            axes[1, 2].plot(df_percentiles['concurrency_numeric'], df_percentiles['p95'], '^-', label='P95', linewidth=2)
            axes[1, 2].set_title(f'Latency Percentiles by Concurrency\n({bucket_method})')
            axes[1, 2].set_xlabel('Concurrency Buckets')
            axes[1, 2].set_ylabel('Latency (seconds)')
            axes[1, 2].set_xticks(df_percentiles['concurrency_numeric'])
            axes[1, 2].set_xticklabels(df_percentiles['concurrency_label'], rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data for percentiles', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Latency Percentiles by Concurrency')
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient concurrency buckets for percentiles', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Latency Percentiles by Concurrency')

    plt.tight_layout()

    # Save the plot
    safe_model_name = model_name.replace("-", "_").replace(".", "_")
    filename = os.path.join(plots_dir, f'concurrency_hypothesis_{safe_model_name}_{timestamp_str}')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')

    if args.show:
        plt.show()

    # Summary insights
    print(f"\n--- Hypothesis Testing Results ---")

    if abs(correlation_exact) > 0.3:
        print(f"✓ HYPOTHESIS SUPPORTED: Strong correlation ({correlation_exact:.3f}) between concurrency and latency")
    elif abs(correlation_exact) > 0.1:
        print(f"? HYPOTHESIS PARTIALLY SUPPORTED: Moderate correlation ({correlation_exact:.3f}) between concurrency and latency")
    else:
        print(f"✗ HYPOTHESIS NOT SUPPORTED: Weak correlation ({correlation_exact:.3f}) between concurrency and latency")

    if len(high_concurrency) > 0 and len(low_concurrency) > 0:
        if performance_impact > 10:
            print(f"✓ SIGNIFICANT PERFORMANCE IMPACT: {performance_impact:+.1f}% latency increase during high concurrency")
        elif performance_impact > 5:
            print(f"? MODERATE PERFORMANCE IMPACT: {performance_impact:+.1f}% latency increase during high concurrency")
        else:
            print(f"✗ MINIMAL PERFORMANCE IMPACT: {performance_impact:+.1f}% latency change during high concurrency")

    print(f"\nPlots saved as:")
    print(f" - {filename}.png")
    print(f" - {filename}.pdf")

    return df_concurrency


def analyze_concurrent_requests(df_model, model_name, bucket_seconds=300, method='start_time'):
    """
    Analyze concurrent request patterns and their impact on response times

    Args:
        df_model: DataFrame with request data
        model_name: Name of the model being analyzed
        bucket_seconds: Size of time buckets in seconds
        method: 'start_time' (faster, use request start time only) or 'overlap' (check overlapping execution)
    """

    print(f"\n{'='*60}")
    print(f"CONCURRENT REQUEST ANALYSIS FOR MODEL: {model_name}")
    print(f"Time bucket size: {bucket_seconds} seconds ({bucket_seconds//60}min {bucket_seconds%60}s)")
    print(f"Bucket method: {method}")
    print(f"{'='*60}")

    if df_model.empty:
        print("No data found for this model.")
        return None

    # Calculate request start and end times
    df_model = df_model.copy()
    df_model['request_start'] = df_model['logging_time'] - pd.to_timedelta(df_model['latency_seconds'], unit='s')
    df_model['request_end'] = df_model['logging_time']

    # Create time buckets
    min_time = df_model['request_start'].min()
    max_time = df_model['request_end'].max()

    # Create bucket boundaries
    bucket_start = min_time.floor(f'{bucket_seconds}s')
    bucket_end = max_time.ceil(f'{bucket_seconds}s')

    # Generate bucket intervals
    bucket_ranges = pd.date_range(start=bucket_start, end=bucket_end, freq=f'{bucket_seconds}s')

    bucket_stats = []

    print(f"Processing {len(bucket_ranges)-1} time buckets...")

    for i in range(len(bucket_ranges) - 1):
        bucket_start_time = bucket_ranges[i]
        bucket_end_time = bucket_ranges[i + 1]

        if method == 'start_time':
            # Brandt's suggestion: For larger buckets, use start time only (faster)
            requests_in_bucket = df_model[
                (df_model['request_start'] >= bucket_start_time) &
                (df_model['request_start'] < bucket_end_time)
                ]
        else:  # method == 'overlap'
            # Original method: Find requests that were EXECUTING during this bucket (overlapping)
            requests_in_bucket = df_model[
                (df_model['request_start'] < bucket_end_time) &
                (df_model['request_end'] > bucket_start_time)
                ]

        if len(requests_in_bucket) > 0:
            # Calculate token sums and means, handling NaN values
            sum_input_tokens = requests_in_bucket['input_tokens'].sum() if requests_in_bucket['input_tokens'].notna().any() else 0
            sum_output_tokens = requests_in_bucket['output_tokens'].sum() if requests_in_bucket['output_tokens'].notna().any() else 0
            sum_total_tokens = requests_in_bucket['total_tokens'].sum() if requests_in_bucket['total_tokens'].notna().any() else 0

            mean_input_tokens = requests_in_bucket['input_tokens'].mean() if requests_in_bucket['input_tokens'].notna().any() else 0
            mean_output_tokens = requests_in_bucket['output_tokens'].mean() if requests_in_bucket['output_tokens'].notna().any() else 0

            bucket_stat = {
                'bucket_start': bucket_start_time,
                'bucket_end': bucket_end_time,
                'request_count': len(requests_in_bucket),  # Changed from 'concurrent_count' for clarity with start_time method
                'min_response_time': requests_in_bucket['latency_seconds'].min(),
                'mean_response_time': requests_in_bucket['latency_seconds'].mean(),
                'max_response_time': requests_in_bucket['latency_seconds'].max(),
                'sum_input_tokens': sum_input_tokens,
                'sum_output_tokens': sum_output_tokens,
                'sum_total_tokens': sum_total_tokens,
                'mean_input_tokens': mean_input_tokens,
                'mean_output_tokens': mean_output_tokens,
            }
            bucket_stats.append(bucket_stat)

    if not bucket_stats:
        print("No bucket data found.")
        return None

    # Convert to DataFrame
    df_buckets = pd.DataFrame(bucket_stats)

    # Print summary statistics
    request_label = "Concurrent requests" if method == 'overlap' else "Requests"
    print(f"\n--- {request_label.title()} Analysis Summary ---")
    print(f"Total time buckets analyzed: {len(df_buckets)}")
    print(f"Bucket size: {bucket_seconds} seconds ({bucket_seconds//60}min {bucket_seconds%60}s)")
    print(f"Method: {method}")
    print(f"{request_label} per bucket:")
    print(f"  Min: {df_buckets['request_count'].min()}")
    print(f"  Mean: {df_buckets['request_count'].mean():.1f}")
    print(f"  Max: {df_buckets['request_count'].max()}")
    print(f"  Median: {df_buckets['request_count'].median():.1f}")

    print(f"\nResponse time statistics:")
    print(f"  Overall min response time: {df_buckets['min_response_time'].min():.3f}s")
    print(f"  Overall mean response time: {df_buckets['mean_response_time'].mean():.3f}s")
    print(f"  Overall max response time: {df_buckets['max_response_time'].max():.3f}s")

    # Correlation analysis
    request_vs_mean_latency = np.corrcoef(df_buckets['request_count'], df_buckets['mean_response_time'])[0, 1]
    request_vs_max_latency = np.corrcoef(df_buckets['request_count'], df_buckets['max_response_time'])[0, 1]

    print(f"\nCorrelation Analysis:")
    print(f"  {request_label} vs Mean Response Time: {request_vs_mean_latency:.3f}")
    print(f"  {request_label} vs Max Response Time: {request_vs_max_latency:.3f}")

    # Token correlation analysis (only if we have token data)
    input_tokens_vs_mean_latency = 0
    output_tokens_vs_mean_latency = 0

    if df_buckets['sum_input_tokens'].sum() > 0:
        input_tokens_vs_mean_latency = np.corrcoef(df_buckets['sum_input_tokens'], df_buckets['mean_response_time'])[0, 1]
        print(f"  Input Tokens vs Mean Response Time: {input_tokens_vs_mean_latency:.3f}")

    if df_buckets['sum_output_tokens'].sum() > 0:
        output_tokens_vs_mean_latency = np.corrcoef(df_buckets['sum_output_tokens'], df_buckets['mean_response_time'])[0, 1]
        print(f"  Output Tokens vs Mean Response Time: {output_tokens_vs_mean_latency:.3f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    bucket_label = f'{bucket_seconds}s ({bucket_seconds//60}min {bucket_seconds%60}s)' if bucket_seconds >= 60 else f'{bucket_seconds}s'
    fig.suptitle(f'{request_label.title()} Analysis - {model_name}\n{bucket_label} buckets, {method} method', fontsize=16, fontweight='bold')

    # Plot 1: Request count over time
    axes[0, 0].plot(df_buckets['bucket_start'], df_buckets['request_count'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_title(f'{request_label.title()} Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel(f'{request_label.title()} Count')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Response times over time
    axes[0, 1].plot(df_buckets['bucket_start'], df_buckets['min_response_time'], label='Min', alpha=0.7, linewidth=1)
    axes[0, 1].plot(df_buckets['bucket_start'], df_buckets['mean_response_time'], label='Mean', linewidth=2)
    axes[0, 1].plot(df_buckets['bucket_start'], df_buckets['max_response_time'], label='Max', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Response Times Over Time')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Response Time (seconds)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Request count vs Mean Response Time
    axes[0, 2].scatter(df_buckets['request_count'], df_buckets['mean_response_time'], alpha=0.6, s=50)
    axes[0, 2].set_title(f'{request_label.title()} vs Mean Response Time\nCorr: {request_vs_mean_latency:.3f}')
    axes[0, 2].set_xlabel(f'{request_label.title()} Count')
    axes[0, 2].set_ylabel('Mean Response Time (seconds)')
    axes[0, 2].grid(True, alpha=0.3)

    # Add trend line
    z, p = safe_polyfit(df_buckets['request_count'], df_buckets['mean_response_time'])
    if z is not None and p is not None:
        axes[0, 2].plot(df_buckets['request_count'], p(df_buckets['request_count']), "r--", alpha=0.8, linewidth=2)

    # Plot 4: Token usage over time
    if df_buckets['sum_input_tokens'].sum() > 0 or df_buckets['sum_output_tokens'].sum() > 0:
        if df_buckets['sum_input_tokens'].sum() > 0:
            axes[1, 0].plot(df_buckets['bucket_start'], df_buckets['sum_input_tokens'], label='Input Tokens', alpha=0.7, linewidth=2)
        if df_buckets['sum_output_tokens'].sum() > 0:
            axes[1, 0].plot(df_buckets['bucket_start'], df_buckets['sum_output_tokens'], label='Output Tokens', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('Token Usage Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Sum of Tokens')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No token data available', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Token Usage Over Time')

    # Plot 5: Input tokens vs Response time
    if df_buckets['sum_input_tokens'].sum() > 0:
        axes[1, 1].scatter(df_buckets['sum_input_tokens'], df_buckets['mean_response_time'], alpha=0.6, color='green', s=50)
        axes[1, 1].set_title(f'Input Tokens vs Mean Response Time\nCorr: {input_tokens_vs_mean_latency:.3f}')
        axes[1, 1].set_xlabel('Sum of Input Tokens')
        axes[1, 1].set_ylabel('Mean Response Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)

        # Add trend line
        z, p = safe_polyfit(df_buckets['sum_input_tokens'], df_buckets['mean_response_time'])
        if z is not None and p is not None:
            axes[1, 1].plot(df_buckets['sum_input_tokens'], p(df_buckets['sum_input_tokens']), "r--", alpha=0.8, linewidth=2)
    else:
        axes[1, 1].text(0.5, 0.5, 'No input token data available', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Input Tokens vs Response Time')

    # Plot 6: Output tokens vs Response time
    if df_buckets['sum_output_tokens'].sum() > 0:
        axes[1, 2].scatter(df_buckets['sum_output_tokens'], df_buckets['mean_response_time'], alpha=0.6, color='orange', s=50)
        axes[1, 2].set_title(f'Output Tokens vs Mean Response Time\nCorr: {output_tokens_vs_mean_latency:.3f}')
        axes[1, 2].set_xlabel('Sum of Output Tokens')
        axes[1, 2].set_ylabel('Mean Response Time (seconds)')
        axes[1, 2].grid(True, alpha=0.3)

        # Add trend line
        z, p = safe_polyfit(df_buckets['sum_output_tokens'], df_buckets['mean_response_time'])
        if z is not None and p is not None:
            axes[1, 2].plot(df_buckets['sum_output_tokens'], p(df_buckets['sum_output_tokens']), "r--", alpha=0.8, linewidth=2)
    else:
        axes[1, 2].text(0.5, 0.5, 'No output token data available', ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Output Tokens vs Response Time')

    plt.tight_layout()

    # Save the plot
    safe_model_name = model_name.replace("-", "_").replace(".", "_")
    method_suffix = "start" if method == 'start_time' else "overlap"
    filename = os.path.join(plots_dir, f'concurrent_analysis_{safe_model_name}_{bucket_seconds}s_{method_suffix}_{timestamp_str}')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')

    if args.show:
        plt.show()

    # Create summary table
    print(f"\n--- Bucket-by-Bucket Analysis (showing top 10 by {request_label.lower()}) ---")
    top_buckets = df_buckets.nlargest(10, 'request_count')

    for _, bucket in top_buckets.iterrows():
        print(f"Time: {bucket['bucket_start'].strftime('%H:%M:%S')} - {bucket['bucket_end'].strftime('%H:%M:%S')}")
        print(f"  {request_label}: {bucket['request_count']}")
        print(f"  Response times - Min: {bucket['min_response_time']:.3f}s, Mean: {bucket['mean_response_time']:.3f}s, Max: {bucket['max_response_time']:.3f}s")
        if bucket['sum_input_tokens'] > 0 or bucket['sum_output_tokens'] > 0:
            print(f"  Tokens - Input: {bucket['sum_input_tokens']:.0f}, Output: {bucket['sum_output_tokens']:.0f}")
        print()

    # Key insights
    print(f"\n--- Key Insights for {model_name} ({bucket_seconds}s buckets, {method} method) ---")

    # Request load insights
    high_load_threshold = df_buckets['request_count'].quantile(0.9)
    high_load_buckets = df_buckets[df_buckets['request_count'] >= high_load_threshold]

    if len(high_load_buckets) > 0:
        avg_response_high_load = high_load_buckets['mean_response_time'].mean()
        avg_response_low_load = df_buckets[df_buckets['request_count'] < high_load_threshold]['mean_response_time'].mean()

        print(f"1. Load Impact:")
        print(f"   - High load (≥{high_load_threshold:.0f} requests): {avg_response_high_load:.3f}s avg response time")
        print(f"   - Low load (<{high_load_threshold:.0f} requests): {avg_response_low_load:.3f}s avg response time")
        if avg_response_low_load > 0:
            print(f"   - Performance impact: {((avg_response_high_load - avg_response_low_load) / avg_response_low_load * 100):+.1f}%")

    # Correlation strength interpretation
    print(f"2. Correlation Analysis:")
    if abs(request_vs_mean_latency) > 0.7:
        strength = "Strong"
    elif abs(request_vs_mean_latency) > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"

    print(f"   - {request_label} vs Response Time: {strength} correlation ({request_vs_mean_latency:.3f})")

    if df_buckets['sum_input_tokens'].sum() > 0:
        if abs(input_tokens_vs_mean_latency) > 0.7:
            token_strength = "Strong"
        elif abs(input_tokens_vs_mean_latency) > 0.3:
            token_strength = "Moderate"
        else:
            token_strength = "Weak"
        print(f"   - Input Tokens vs Response Time: {token_strength} correlation ({input_tokens_vs_mean_latency:.3f})")

    print(f"\nPlots saved as:")
    print(f" - {filename}.png")
    print(f" - {filename}.pdf")

    return df_buckets

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
    if 'output_tokens' in df_model.columns and df_model['output_tokens'].notna().any():
        df_tokens = df_model.dropna(subset=['output_tokens'])
        if len(df_tokens) > 0:
            print(f"Token statistics:")
            print(f"  Mean output tokens: {df_tokens['output_tokens'].mean():.1f}")
            print(f"  Median output tokens: {df_tokens['output_tokens'].median():.1f}")
            print(f"  Min output tokens: {df_tokens['output_tokens'].min()}")
            print(f"  Max output tokens: {df_tokens['output_tokens'].max()}")

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
        token_latency_corr = np.corrcoef(df_tokens['output_tokens'], df_tokens['latency_seconds'])[0, 1]
        if abs(token_latency_corr) > 0.7:
            correlation_strength = "Strong"
        elif abs(token_latency_corr) > 0.3:
            correlation_strength = "Moderate"
        else:
            correlation_strength = "Weak"

    # Create comprehensive visualizations
    plt.style.use('default')
    fig = plt.figure(figsize=(32, 36))  # Increased height for additional plots

    # Add page header
    generation_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    add_page_header(fig, project_id, start_filter_timestamp, end_filter_timestamp, generation_time,
                    f" - Model Analysis: {model_name}")

    fig.suptitle(f'Gemini Log Analysis - {model_name}',
                 fontsize=18, fontweight='bold', y=0.94)

    # === ROW 1: SUMMARY TABLES (3 tables across the full width) ===

    # Table 1: Basic Statistics (ENHANCED WITH STD DEV DATA)
    ax1 = plt.subplot(5, 3, 1)  # Changed to 5 rows
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
    ax2 = plt.subplot(5, 3, 2)
    ax2.axis('tight')
    ax2.axis('off')

    if df_tokens is not None and len(df_tokens) > 0:
        token_stats = [
            ['Metric', 'Value'],
            ['Mean Output Tokens', f'{df_tokens["output_tokens"].mean():.1f}'],
            ['Median Output Tokens', f'{df_tokens["output_tokens"].median():.1f}'],
            ['Min Output Tokens', f'{df_tokens["output_tokens"].min():,}'],
            ['Max Output Tokens', f'{df_tokens["output_tokens"].max():,}'],
            ['Token-Latency Corr', f'{token_latency_corr:.3f}' if token_latency_corr is not None else 'N/A'],
            ['Correlation Strength', correlation_strength],
        ]
    else:
        token_stats = [
            ['Metric', 'Value'],
            ['Mean Output Tokens', 'N/A'],
            ['Median Output Tokens', 'N/A'],
            ['Min Output Tokens', 'N/A'],
            ['Max Output Tokens', 'N/A'],
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
    ax3 = plt.subplot(5, 3, 3)
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
    ax1 = plt.subplot(5, 3, 4)
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
    ax6 = plt.subplot(5, 3, 5)
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

    # Plot 3: Latency vs Token Count (Original - Linear Scale) with smaller dots
    ax5 = plt.subplot(5, 3, 6)
    if df_tokens is not None and len(df_tokens) > 0:
        # Use input tokens for color if available, otherwise use latency
        if df_tokens['input_tokens'].notna().any():
            color_data = df_tokens['input_tokens']
            color_label = 'Input Tokens'
            cmap = 'plasma'
        else:
            color_data = df_tokens['latency_seconds']
            color_label = 'Latency (seconds)'
            cmap = 'viridis'

        scatter = plt.scatter(df_tokens['output_tokens'], df_tokens['latency_seconds'],
                              alpha=0.6, s=8, c=color_data,  # Smaller dots (s=8)
                              cmap=cmap, edgecolors='black', linewidth=0.1)

        plt.colorbar(scatter, label=color_label)
        plt.title('Latency vs Output Token Count (Linear)', fontsize=14, fontweight='bold')
        plt.xlabel('Output Token Count')
        plt.ylabel('Latency (seconds)')
        plt.grid(True, alpha=0.3)

        if token_latency_corr is not None:
            plt.text(0.05, 0.95, f'Correlation: {token_latency_corr:.3f}',
                     transform=ax5.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Safe trend line
            z, p = safe_polyfit(df_tokens['output_tokens'], df_tokens['latency_seconds'])
            if z is not None and p is not None:
                plt.plot(df_tokens['output_tokens'], p(df_tokens['output_tokens']),
                         "r--", alpha=0.8, linewidth=2)
    else:
        plt.text(0.5, 0.5, 'No token count data available',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        plt.title('Latency vs Token Count (Linear)', fontsize=14, fontweight='bold')

    # === ROW 3: NEW TOKEN ANALYSIS WITH LOG SCALE ===

    # NEW Plot: Latency vs Token Count (Log Scale) with smaller dots
    ax_log = plt.subplot(5, 3, 7)
    if df_tokens is not None and len(df_tokens) > 0:
        # Filter out zero or negative token counts for log scale
        df_tokens_positive = df_tokens[df_tokens['output_tokens'] > 0]

        if len(df_tokens_positive) > 0:
            # Use input tokens for color if available
            if df_tokens_positive['input_tokens'].notna().any():
                color_data = df_tokens_positive['input_tokens']
                color_label = 'Input Tokens'
                cmap = 'plasma'
            else:
                color_data = df_tokens_positive['latency_seconds']
                color_label = 'Latency (seconds)'
                cmap = 'viridis'

            scatter = plt.scatter(df_tokens_positive['output_tokens'], df_tokens_positive['latency_seconds'],
                                  alpha=0.6, s=8, c=color_data,  # Smaller dots (s=8)
                                  cmap=cmap, edgecolors='black', linewidth=0.1)

            plt.colorbar(scatter, label=color_label)
            plt.xscale('log')  # This is the key change - logarithmic x-axis
            plt.title('Latency vs Output Token Count (Log Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Output Token Count (Log Scale)')
            plt.ylabel('Latency (seconds)')
            plt.grid(True, alpha=0.3)

            # Add correlation info
            if token_latency_corr is not None:
                plt.text(0.05, 0.95, f'Correlation: {token_latency_corr:.3f}',
                         transform=ax_log.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Add trend line on log scale
            log_tokens = np.log10(df_tokens_positive['output_tokens'])
            z, p = safe_polyfit(log_tokens, df_tokens_positive['latency_seconds'])
            if z is not None and p is not None:
                sorted_tokens = np.sort(df_tokens_positive['output_tokens'])
                sorted_log_tokens = np.log10(sorted_tokens)
                plt.plot(sorted_tokens, p(sorted_log_tokens), "r--", alpha=0.8, linewidth=2)

            # Add some helpful annotations for token ranges
            token_min = df_tokens_positive['output_tokens'].min()
            token_max = df_tokens_positive['output_tokens'].max()
            plt.text(0.05, 0.85, f'Range: {token_min}-{token_max} tokens',
                     transform=ax_log.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'No positive token count data available',
                     ha='center', va='center', transform=ax_log.transAxes, fontsize=12)
            plt.title('Latency vs Token Count (Log Scale)', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No token count data available',
                 ha='center', va='center', transform=ax_log.transAxes, fontsize=12)
        plt.title('Latency vs Token Count (Log Scale)', fontsize=14, fontweight='bold')

    # Plot 7: Cumulative distribution
    ax10 = plt.subplot(5, 3, 8)
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

    # Token Count Distribution
    ax9 = plt.subplot(5, 3, 9)
    if df_tokens is not None and len(df_tokens) > 0:
        plt.hist(df_tokens['output_tokens'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Output Token Count Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Output Token Count')
        plt.ylabel('Frequency')
        plt.axvline(df_tokens['output_tokens'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df_tokens["output_tokens"].mean():.1f}')
        plt.axvline(df_tokens['output_tokens'].median(), color='green', linestyle='--',
                    label=f'Median: {df_tokens["output_tokens"].median():.1f}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No token count data', ha='center', va='center', transform=ax9.transAxes)
        plt.title('Token Count Distribution', fontsize=14, fontweight='bold')

    # === ROW 4: DETAILED ANALYSIS ===

    # Plot 5: Load Test Sequence
    ax8 = plt.subplot(5, 3, 10)
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
    plt.xlabel('Request Number')
    plt.ylabel('Latency (seconds)')
    plt.grid(True, alpha=0.3)

    plt.text(0.02, 0.98, f'Total: {len(df_sorted)} requests',
             transform=ax8.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Empty plot for spacing
    ax11 = plt.subplot(5, 3, 11)
    ax11.axis('off')

    # Box plot by hour
    ax12 = plt.subplot(5, 3, 12)
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

    # === ROW 5: TOKEN ANALYSIS DETAILS ===

    # Token vs Latency Binned Analysis
    ax13 = plt.subplot(5, 3, 13)
    if df_tokens is not None and len(df_tokens) > 0:
        # Create token bins for analysis
        df_tokens['token_bin'] = pd.cut(df_tokens['output_tokens'], bins=10, precision=0)
        token_bin_stats = df_tokens.groupby('token_bin')['latency_seconds'].agg(['mean', 'count']).reset_index()

        # Only show bins with reasonable sample sizes
        token_bin_stats = token_bin_stats[token_bin_stats['count'] >= 5]

        if len(token_bin_stats) > 0:
            bin_centers = [interval.mid for interval in token_bin_stats['token_bin']]
            plt.bar(range(len(bin_centers)), token_bin_stats['mean'], alpha=0.7, color='lightgreen')
            plt.title('Mean Latency by Token Count Bins', fontsize=14, fontweight='bold')
            plt.xlabel('Token Count Bins')
            plt.ylabel('Mean Latency (seconds)')
            plt.xticks(range(len(bin_centers)), [f'{int(x)}' for x in bin_centers], rotation=45)

            # Add count labels on bars
            for i, (mean_lat, count) in enumerate(zip(token_bin_stats['mean'], token_bin_stats['count'])):
                plt.text(i, mean_lat + 0.01, f'n={count}', ha='center', va='bottom', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for binning', ha='center', va='center', transform=ax13.transAxes)
            plt.title('Mean Latency by Token Count Bins', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No token data available', ha='center', va='center', transform=ax13.transAxes)
        plt.title('Mean Latency by Token Count Bins', fontsize=14, fontweight='bold')

    # Token Count vs Request Order
    ax14 = plt.subplot(5, 3, 14)
    if df_tokens is not None and len(df_tokens) > 0:
        df_tokens_sorted = df_tokens.sort_values('logging_time').reset_index(drop=True)
        df_tokens_sorted['request_order'] = range(1, len(df_tokens_sorted) + 1)

        plt.scatter(df_tokens_sorted['request_order'], df_tokens_sorted['output_tokens'],
                    alpha=0.6, s=20, c=df_tokens_sorted['latency_seconds'], cmap='viridis')
        plt.colorbar(label='Latency (seconds)')
        plt.title('Token Count Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Request Order')
        plt.ylabel('Output Token Count')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No token data available', ha='center', va='center', transform=ax14.transAxes)
        plt.title('Token Count Over Time', fontsize=14, fontweight='bold')

    # Token Count Distribution (Log Scale)
    ax15 = plt.subplot(5, 3, 15)
    if df_tokens is not None and len(df_tokens) > 0:
        df_tokens_positive = df_tokens[df_tokens['output_tokens'] > 0]
        if len(df_tokens_positive) > 0:
            plt.hist(df_tokens_positive['output_tokens'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xscale('log')
            plt.title('Output Token Distribution (Log Scale)', fontsize=14, fontweight='bold')
            plt.xlabel('Output Token Count (Log Scale)')
            plt.ylabel('Frequency')
            plt.axvline(df_tokens_positive['output_tokens'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df_tokens_positive["output_tokens"].mean():.1f}')
            plt.axvline(df_tokens_positive['output_tokens'].median(), color='green', linestyle='--',
                        label=f'Median: {df_tokens_positive["output_tokens"].median():.1f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No positive token data', ha='center', va='center', transform=ax15.transAxes)
            plt.title('Token Distribution (Log Scale)', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No token data available', ha='center', va='center', transform=ax15.transAxes)
        plt.title('Token Distribution (Log Scale)', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.90], pad=4.0)

    # Save plot to file
    safe_model_name = model_name.replace("-", "_").replace(".", "_")
    filename = os.path.join(plots_dir, f'gemini_analysis_{safe_model_name}_{timestamp_str}')

    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', facecolor='white')

    if args.show:
        plt.show()

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

        # Token range analysis
        token_min = df_tokens['output_tokens'].min()
        token_max = df_tokens['output_tokens'].max()
        token_median = df_tokens['output_tokens'].median()
        print(f"   - Token range: {token_min} to {token_max} (median: {token_median:.0f})")

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

    # Extract token counts and model name
    print("Extracting token counts and model names...")
    token_data = df_gemini['full_response_json'].apply(extract_token_counts)
    df_gemini[['input_tokens', 'output_tokens', 'total_tokens']] = pd.DataFrame(token_data.tolist(), index=df_gemini.index)
    df_gemini['model_name'] = df_gemini['model'].apply(extract_model_name)

    # Print overall summary
    print(f"\n--- OVERALL SUMMARY ---")
    print(f"Total requests: {len(df_gemini)}")
    print(f"Requests with input token data: {df_gemini['input_tokens'].notna().sum()}")
    print(f"Requests with output token data: {df_gemini['output_tokens'].notna().sum()}")
    print(f"Unique models: {df_gemini['model_name'].nunique()}")
    print(f"Models found: {sorted(df_gemini['model_name'].unique())}")

    # Store generation time for consistent headers
    generation_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Analyze each model separately
    for model_name in sorted(df_gemini['model_name'].unique()):
        if pd.notna(model_name):
            df_model = df_gemini[df_gemini['model_name'] == model_name].copy()

            # Run original analysis (enhanced with new token plots)
            analyze_model_data(df_model, model_name)

            # NEW: Create standalone high-resolution latency vs token plots
            df_tokens = df_model.dropna(subset=['output_tokens']) if 'output_tokens' in df_model.columns else None
            if df_tokens is not None and len(df_tokens) > 0:
                create_latency_token_plots(df_tokens, model_name, start_filter_timestamp,
                                           end_filter_timestamp, generation_time)

            # NEW: Create enhanced hourly analysis with weekday differentiation
            create_hourly_analysis_with_weekday(df_model, model_name, start_filter_timestamp,
                                                end_filter_timestamp, generation_time)

            # Run concurrent analysis with configurable bucket sizes
            for bucket_size in bucket_sizes:
                print(f"\n" + "="*80)
                print(f"ANALYZING {model_name} with {bucket_size}-second buckets")
                print("="*80)
                analyze_concurrent_requests(df_model, model_name, bucket_size, bucket_method)

            # Run focused concurrency hypothesis testing
            print(f"\n" + "="*80)
            print(f"TESTING CONCURRENCY HYPOTHESIS FOR {model_name}")
            print("="*80)
            analyze_concurrency_hypothesis(df_model, model_name)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()