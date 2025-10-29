import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_latency_breakdown_new_3_bars(
        csv_path,
        figsize=(5.0, 2.5),
        save_path=None,
        dpi=300,
        min_x_value=None,
        max_x_value=None,
        take_avg=False,
        x_flops=False,
        x_sizes=False,
        x_descending=False,
        show_legend=True,
        y_scientific=False,
        y_max=None):
    """
    Create IEEE conference paper style bar plot for latency breakdown.

    Shows overall latency as full bars with data flow cycles overlaid on top
    to visualize the proportion of time spent on data communication.
    Note: comp_cycles and data_flow_cycles overlap, so they don't sum to overall_latency_cycles.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with latency/energy results
    figsize : tuple
        Figure size in inches (default: 5.0x2.5 for wider plot)
    save_path : str, optional
        Path to save the figure (if None, only displays)
    dpi : int
        Resolution for saved figure (default: 300)
    min_x_value : float, optional
        Minimum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    max_x_value : float, optional
        Maximum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    take_avg : bool
        If True, average all metrics for duplicate parts_per_layer values (only when x_flops and x_sizes are False)
    x_flops : bool
        If True, use target_value (FLOPs per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
    x_sizes : bool
        If True, use target_value (Size in KB per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
        Note: Only one of x_flops or x_sizes should be True
    x_descending : bool
        If True, sort x-axis in descending order (default: False)
    show_legend : bool
        If True, display the legend (default: True)
    y_scientific : bool
        If True, use scientific notation for y-axis (e.g., 1e6) (default: False)
    y_max : float, optional
        Set maximum value for y-axis. If None, uses automatic scaling

    Returns:
    --------
    fig, ax : matplotlib objects
    """

    # Set Helvetica font and IEEE style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # Read data
    df = pd.read_csv(csv_path)
    print(f"Original data: {len(df)} rows")

    # When using x_flops or x_sizes, skip grouping by parts_per_layer unless take_avg is explicitly True
    if (x_flops or x_sizes) and not take_avg:
        # Don't group by parts_per_layer, keep all data points
        df_filtered = df.copy()
    elif take_avg:
        # Average duplicate parts_per_layer entries - specify which columns to average
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Remove parts_per_layer from aggregation since we're grouping by it
        agg_cols = [col for col in numeric_cols if col != 'parts_per_layer']

        # Create aggregation dict: mean for numeric columns
        agg_dict = {col: 'mean' for col in agg_cols}

        # Keep target_value if it exists (take first or mean)
        if 'target_value' in df.columns and 'target_value' not in agg_dict:
            agg_dict['target_value'] = 'first'

        df_filtered = df.groupby('parts_per_layer').agg(agg_dict).reset_index()
    else:
        # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
        df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]

    # ------------------------------------------------------------------ #
    # Determine which column to use for filtering
    # ------------------------------------------------------------------ #
    if (x_flops or x_sizes) and 'target_value' in df_filtered.columns:
        filter_column = 'target_value'
    else:
        filter_column = 'parts_per_layer'

    # Apply minimum/maximum filters on the appropriate column
    if min_x_value is not None:
        df_filtered = df_filtered[df_filtered[filter_column] >= min_x_value]
        print(f"Filtering: showing only {filter_column} >= {min_x_value}")
    if max_x_value is not None:
        df_filtered = df_filtered[df_filtered[filter_column] <= max_x_value]
        print(f"Filtering: showing only {filter_column} <= {max_x_value}")

    # Determine sorting key based on x-axis choice
    if x_flops and 'target_value' in df_filtered.columns:
        sort_key = 'target_value'
    elif x_sizes and 'target_value' in df_filtered.columns:
        sort_key = 'target_value'
    else:
        sort_key = 'parts_per_layer'

    # Sort by the appropriate key (ascending or descending)
    df_sorted = df_filtered.sort_values(sort_key, ascending=not x_descending)

    print(f"After filtering and processing: {len(df_sorted)} rows")

    # Calculate data flow percentage
    df_sorted = df_sorted.copy()
    df_sorted['data_flow_percentage'] = (df_sorted['data_flow_cycles'] /
                                         df_sorted['overall_latency_cycles'] * 100)

    # ------------------------------------------------------------------ #
    # Determine x-axis values and labels
    # ------------------------------------------------------------------ #
    if x_flops and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value']
        x_label = 'FLOPs per Partition'
        # Format FLOPs values (convert to K or M if large)
        x_tick_labels = [f"{int(v/1000)}K" if v >= 1000 else str(int(v)) for v in x_values]
    elif x_sizes and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value'] / 1024  # Convert to KB
        x_label = 'Size per Partition (KB)'
        x_tick_labels = [f"{int(v)}" for v in x_values]
    else:
        x_values = df_sorted['parts_per_layer']
        x_label = 'Partitions per Layer'
        x_tick_labels = df_sorted['parts_per_layer'].astype(int)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    x_positions = np.arange(len(df_sorted))
    outer_width = 0.85
    inner_width = outer_width * 0.50
    left_offset  = -inner_width / 2.0
    right_offset = +inner_width / 2.0

    # Colors (keep the ones you defined)
    color_overall = "#F76D55"   # red
    color_dataflow = "#4A83EC"  # blue

    # 2. Left inner bar – Computation
    comp_cycles = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
    ax.bar([p + left_offset for p in x_positions],
            comp_cycles,
            width=inner_width,
            color=color_overall,
            edgecolor='black',
            linewidth=0.5,
            label='Comp PE',
            zorder=3)

    # 3. Right inner bar – Data Flow
    ax.bar([p + right_offset for p in x_positions],
            df_sorted['data_flow_cycles'],
            width=inner_width,
            color=color_dataflow,
            edgecolor='black',
            linewidth=0.5,
            label='Comm Flow',
            zorder=3,
            alpha=1)

    # 1. Outer empty bar
    ax.bar(x_positions,
            df_sorted['overall_latency_cycles'],
            width=outer_width,
            color='none',
            edgecolor='black',
            linewidth=0.65,
            hatch='///',
            label='All',
            zorder=2)

    # ------------------------------------------------------------------
    #   Formatting
    # ------------------------------------------------------------------
    ax.set_xlabel(x_label, fontweight='normal')
    ax.set_ylabel('Latency (cycles)', fontweight='normal')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45)

    # Set y-axis limits
    if y_max is not None:
        ax.set_ylim(0, y_max)

    # Scientific notation for y-axis
    if y_scientific:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)

    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', direction='in')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)

    # Legend
    if show_legend:
        ax.legend(loc='upper center', ncol = 3, frameon=True, framealpha=1.0,
                    edgecolor='black', fancybox=False, facecolor='white', bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_latency_breakdown_with_error(
        csv_path1,                     # <-- latency breakdown file
        csv_path2,                     # <-- error / comparison file
        figsize=(5.0, 2.5),
        save_path=None,
        dpi=300,
        min_x_value=None,
        max_x_value=None,
        take_avg=False,
        x_flops=False,
        x_sizes=False,
        x_descending=False,
        show_legend=True,
        y_scientific=False,
        y_max=None):
    """
    IEEE-style bar plot of latency breakdown (from csv_path1)
    + scatter overlay of analytical error (%) from csv_path2.

    Files are merged on 'parts_per_layer' by default, or on both 'parts_per_layer'
    and 'target_value' when x_flops or x_sizes is True.

    Parameters:
    -----------
    csv_path1 : str
        Path to latency breakdown CSV file
    csv_path2 : str
        Path to error/comparison CSV file
    figsize : tuple
        Figure size in inches (default: 5.0x2.5)
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure (default: 300)
    min_x_value : float, optional
        Minimum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    max_x_value : float, optional
        Maximum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    take_avg : bool
        If True, average all metrics for duplicate parts_per_layer values
    x_flops : bool
        If True, use target_value (FLOPs per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
    x_sizes : bool
        If True, use target_value (Size in KB per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
        Note: Only one of x_flops or x_sizes should be True
    x_descending : bool
        If True, sort x-axis in descending order (default: False)
    show_legend : bool
        If True, display the legend (default: True)
    y_scientific : bool
        If True, use scientific notation for y-axis (e.g., 1e6) (default: False)
    y_max : float, optional
        Set maximum value for y-axis. If None, uses automatic scaling

    Returns:
    --------
    fig, ax1, ax2 : matplotlib objects
    """
    # ------------------------------------------------------------------ #
    # 1.  Style (Helvetica + IEEE)
    # ------------------------------------------------------------------ #
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # ------------------------------------------------------------------ #
    # 2.  Load both CSVs
    # ------------------------------------------------------------------ #
    df1 = pd.read_csv(csv_path1)   # latency breakdown
    df2 = pd.read_csv(csv_path2)   # error metrics

    print(f"Loaded {len(df1)} rows from latency CSV")
    print(f"Loaded {len(df2)} rows from error CSV")

    # ------------------------------------------------------------------ #
    # 3.  Process df1 (latency)
    # ------------------------------------------------------------------ #
    # When using x_flops or x_sizes, skip grouping by parts_per_layer unless take_avg is explicitly True
    if (x_flops or x_sizes) and not take_avg:
        # Don't group by parts_per_layer, keep all data points
        df1_grp = df1.copy()
    elif take_avg:
        num_cols = df1.select_dtypes(include='number').columns.tolist()
        agg_cols = [c for c in num_cols if c != 'parts_per_layer']
        agg_dict = {c: 'mean' for c in agg_cols}
        df1_grp = df1.groupby('parts_per_layer').agg(agg_dict).reset_index()
    else:
        df1_grp = df1.loc[df1.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]

    # ------------------------------------------------------------------ #
    # 4.  Process df2 (error) – average duplicates
    # ------------------------------------------------------------------ #
    # Include target_value in aggregation if it exists
    agg_dict_df2 = {'percentage_diff': 'mean'}
    if 'target_value' in df2.columns:
        agg_dict_df2['target_value'] = 'first'  # or 'mean' if values differ

    # When using x_flops or x_sizes, we may want to keep all df2 rows too
    if (x_flops or x_sizes) and not take_avg:
        df2_grp = df2.copy()
    else:
        df2_grp = df2.groupby('parts_per_layer').agg(agg_dict_df2).reset_index()

    # ------------------------------------------------------------------ #
    # 5.  Merge on appropriate columns
    # ------------------------------------------------------------------ #
    # When using x_flops or x_sizes, merge on both parts_per_layer AND target_value
    # to avoid unwanted grouping
    if (x_flops or x_sizes) and 'target_value' in df1_grp.columns and 'target_value' in df2_grp.columns:
        merge_keys = ['parts_per_layer', 'target_value']
        print(f"Merging on: {merge_keys}")
    else:
        merge_keys = 'parts_per_layer'
        print(f"Merging on: {merge_keys}")

    df_merged = pd.merge(df1_grp, df2_grp, on=merge_keys, how='inner')

    if df_merged.empty:
        raise ValueError(f"No overlapping '{merge_keys}' values between the two CSVs!")

    print(f"After merge: {len(df_merged)} rows")

    # ------------------------------------------------------------------ #
    # 6.  Determine which column to use for filtering
    # ------------------------------------------------------------------ #
    if (x_flops or x_sizes) and 'target_value' in df_merged.columns:
        filter_column = 'target_value'
    else:
        filter_column = 'parts_per_layer'

    # Apply min/max filters on the appropriate column
    if min_x_value is not None:
        df_merged = df_merged[df_merged[filter_column] >= min_x_value]
        print(f"Filtering: showing only {filter_column} >= {min_x_value}")
    if max_x_value is not None:
        df_merged = df_merged[df_merged[filter_column] <= max_x_value]
        print(f"Filtering: showing only {filter_column} <= {max_x_value}")

    # Determine sorting key based on x-axis choice
    if x_flops and 'target_value' in df_merged.columns:
        sort_key = 'target_value'
    elif x_sizes and 'target_value' in df_merged.columns:
        sort_key = 'target_value'
    else:
        sort_key = 'parts_per_layer'

    # Sort by the appropriate key (ascending or descending)
    df_sorted = df_merged.sort_values(sort_key, ascending=not x_descending).reset_index(drop=True)
    print(f"After filter: {len(df_sorted)} configurations")

    # ------------------------------------------------------------------ #
    # 7.  Derived columns
    # ------------------------------------------------------------------ #
    df_sorted['data_flow_percentage'] = (
        df_sorted['data_flow_cycles'] / df_sorted['overall_latency_cycles'] * 100
    )

    # ------------------------------------------------------------------ #
    # 8.  Determine x-axis values and labels
    # ------------------------------------------------------------------ #
    if x_flops and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value']
        x_label = 'FLOPs per Partition'
        # Format FLOPs values (convert to K or M if large)
        x_tick_labels = [f"{int(v/1000)}K" if v >= 1000 else str(int(v)) for v in x_values]
    elif x_sizes and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value'] / 1024  # Convert to KB
        x_label = 'Size per Partition (KB)'
        x_tick_labels = [f"{int(v)}" for v in x_values]
    else:
        x_values = df_sorted['parts_per_layer']
        x_label = 'Partitions per Layer'
        x_tick_labels = df_sorted['parts_per_layer'].astype(int)

    # ------------------------------------------------------------------ #
    # 9.  Plot setup
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    ax2 = ax1.twinx()  # secondary axis for error

    x_pos = np.arange(len(df_sorted))

    # ------------------------------------------------------------------ #
    # 10.  Bars (same as original)
    # ------------------------------------------------------------------ #
    outer_w = 0.85
    inner_w = outer_w * 0.50
    left_off  = -inner_w / 2.0
    right_off = +inner_w / 2.0

    col_comp = "#F76D55"      # red
    col_comm = "#4A83EC"      # blue

    # Computation bar
    comp = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
    ax1.bar([p + left_off  for p in x_pos], comp,
            width=inner_w, color=col_comp, edgecolor='black', linewidth=0.5,
            label='Comp PE', zorder=3)

    # Data-flow bar
    ax1.bar([p + right_off for p in x_pos], df_sorted['data_flow_cycles'],
            width=inner_w, color=col_comm, edgecolor='black', linewidth=0.5,
            label='Comm Flow', zorder=3)

    # Outer outline bar
    ax1.bar(x_pos, df_sorted['overall_latency_cycles'],
            width=outer_w, color='none', edgecolor='black', linewidth=0.65,
            hatch='///', label='All', zorder=2)

    # ------------------------------------------------------------------ #
    # 11. Scatter: Error (%) from csv_path2
    # ------------------------------------------------------------------ #
    err_color = "#126F12"  # green
    ax2.scatter(x_pos, df_sorted['percentage_diff'],
                color=err_color, alpha=0.65, marker='^', s=50, edgecolor='black', linewidth=0.75,
                label='Err', zorder=4)

    # ------------------------------------------------------------------ #
    # 12. Labels, ticks, grid
    # ------------------------------------------------------------------ #
    ax1.set_xlabel(x_label, fontweight='normal')
    ax1.set_ylabel('Latency (cycles)', fontweight='normal')
    ax2.set_ylabel('Error (%)', fontweight='normal')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_tick_labels, rotation=45)

    # Set y-axis limits
    if y_max is not None:
        ax1.set_ylim(0, y_max)

    # Scientific notation for y-axis
    if y_scientific:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax1.yaxis.set_major_formatter(formatter)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(top=False, right=False)

    # ------------------------------------------------------------------ #
    # 13. Legend
    # ------------------------------------------------------------------ #
    if show_legend:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(h1 + h2, l1 + l2,
                loc='upper center', ncol=4, frameon=True, framealpha=1.0,
                edgecolor='black', fancybox=False, facecolor='white',
                bbox_to_anchor=(0.5, 1.05))
        # Ensure legend is drawn on top and not clipped
        legend.set_zorder(100)

    # Use tight_layout with rect to leave space for legend
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved → {save_path}")

    return fig, ax1, ax2

def plot_latency_breakdown_with_trend(
        csv_path1,                     # <-- latency breakdown file
        csv_path2,                     # <-- comparison file
        figsize=(5.0, 2.5),
        save_path=None,
        dpi=300,
        min_x_value=None,
        max_x_value=None,
        take_avg=False,
        x_flops=False,
        x_sizes=False,
        x_descending=False,
        show_legend=True,
        y_scientific=False,
        y_max=None):
    """
    IEEE-style bar plot of latency breakdown (from csv_path1)
    + scatter overlay of analytical error (%) from csv_path2.

    Files are merged on 'parts_per_layer' by default, or on both 'parts_per_layer'
    and 'target_value' when x_flops or x_sizes is True.

    Parameters:
    -----------
    csv_path1 : str
        Path to latency breakdown CSV file
    csv_path2 : str
        Path to error/comparison CSV file
    figsize : tuple
        Figure size in inches (default: 5.0x2.5)
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saved figure (default: 300)
    min_x_value : float, optional
        Minimum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    max_x_value : float, optional
        Maximum value for x-axis filtering (applies to target_value when x_flops/x_sizes=True,
        or parts_per_layer otherwise)
    take_avg : bool
        If True, average all metrics for duplicate parts_per_layer values
    x_flops : bool
        If True, use target_value (FLOPs per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
    x_sizes : bool
        If True, use target_value (Size in KB per partition) for x-axis instead of parts_per_layer
        When True, no grouping by parts_per_layer is performed unless take_avg=True
        Note: Only one of x_flops or x_sizes should be True
    x_descending : bool
        If True, sort x-axis in descending order (default: False)
    show_legend : bool
        If True, display the legend (default: True)
    y_scientific : bool
        If True, use scientific notation for y-axis (e.g., 1e6) (default: False)
    y_max : float, optional
        Set maximum value for y-axis. If None, uses automatic scaling

    Returns:
    --------
    fig, ax1, ax2 : matplotlib objects
    """
    # ------------------------------------------------------------------ #
    # 1.  Style (Helvetica + IEEE)
    # ------------------------------------------------------------------ #
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # ------------------------------------------------------------------ #
    # 2.  Load both CSVs
    # ------------------------------------------------------------------ #
    df1 = pd.read_csv(csv_path1)   # latency breakdown
    df2 = pd.read_csv(csv_path2)   # error metrics

    print(f"Loaded {len(df1)} rows from latency CSV")
    print(f"Loaded {len(df2)} rows from error CSV")

    # ------------------------------------------------------------------ #
    # 3.  Process df1 (latency)
    # ------------------------------------------------------------------ #
    # When using x_flops or x_sizes, skip grouping by parts_per_layer unless take_avg is explicitly True
    if (x_flops or x_sizes) and not take_avg:
        # Don't group by parts_per_layer, keep all data points
        df1_grp = df1.copy()
    elif take_avg:
        num_cols = df1.select_dtypes(include='number').columns.tolist()
        agg_cols = [c for c in num_cols if c != 'parts_per_layer']
        agg_dict = {c: 'mean' for c in agg_cols}
        df1_grp = df1.groupby('parts_per_layer').agg(agg_dict).reset_index()
    else:
        df1_grp = df1.loc[df1.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]

    # ------------------------------------------------------------------ #
    # 4.  Process df2 (error) – average duplicates
    # ------------------------------------------------------------------ #
    # Include target_value in aggregation if it exists
    agg_dict_df2 = {'percentage_diff': 'mean'}
    if 'target_value' in df2.columns:
        agg_dict_df2['target_value'] = 'first'  # or 'mean' if values differ

    # When using x_flops or x_sizes, we may want to keep all df2 rows too
    if (x_flops or x_sizes) and not take_avg:
        df2_grp = df2.copy()
    else:
        df2_grp = df2.groupby('parts_per_layer').agg(agg_dict_df2).reset_index()

    # ------------------------------------------------------------------ #
    # 5.  Merge on appropriate columns
    # ------------------------------------------------------------------ #
    # When using x_flops or x_sizes, merge on both parts_per_layer AND target_value
    # to avoid unwanted grouping
    if (x_flops or x_sizes) and 'target_value' in df1_grp.columns and 'target_value' in df2_grp.columns:
        merge_keys = ['parts_per_layer', 'target_value']
        print(f"Merging on: {merge_keys}")
    else:
        merge_keys = 'parts_per_layer'
        print(f"Merging on: {merge_keys}")

    df_merged = pd.merge(df1_grp, df2_grp, on=merge_keys, how='inner')

    if df_merged.empty:
        raise ValueError(f"No overlapping '{merge_keys}' values between the two CSVs!")

    print(f"After merge: {len(df_merged)} rows")

    # ------------------------------------------------------------------ #
    # 6.  Determine which column to use for filtering
    # ------------------------------------------------------------------ #
    if (x_flops or x_sizes) and 'target_value' in df_merged.columns:
        filter_column = 'target_value'
    else:
        filter_column = 'parts_per_layer'

    # Apply min/max filters on the appropriate column
    if min_x_value is not None:
        df_merged = df_merged[df_merged[filter_column] >= min_x_value]
        print(f"Filtering: showing only {filter_column} >= {min_x_value}")
    if max_x_value is not None:
        df_merged = df_merged[df_merged[filter_column] <= max_x_value]
        print(f"Filtering: showing only {filter_column} <= {max_x_value}")

    # Determine sorting key based on x-axis choice
    if x_flops and 'target_value' in df_merged.columns:
        sort_key = 'target_value'
    elif x_sizes and 'target_value' in df_merged.columns:
        sort_key = 'target_value'
    else:
        sort_key = 'parts_per_layer'

    # Sort by the appropriate key (ascending or descending)
    df_sorted = df_merged.sort_values(sort_key, ascending=not x_descending).reset_index(drop=True)
    print(f"After filter: {len(df_sorted)} configurations")

    # ------------------------------------------------------------------ #
    # 7.  Derived columns
    # ------------------------------------------------------------------ #
    df_sorted['data_flow_percentage'] = (
        df_sorted['data_flow_cycles'] / df_sorted['overall_latency_cycles'] * 100
    )

    # ------------------------------------------------------------------ #
    # 8.  Determine x-axis values and labels
    # ------------------------------------------------------------------ #
    if x_flops and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value']
        x_label = 'FLOPs per Partition'
        # Format FLOPs values (convert to K or M if large)
        x_tick_labels = [f"{int(v/1000)}K" if v >= 1000 else str(int(v)) for v in x_values]
    elif x_sizes and 'target_value' in df_sorted.columns:
        x_values = df_sorted['target_value'] / 1024  # Convert to KB
        x_label = 'Size per Partition (KB)'
        x_tick_labels = [f"{int(v)}" for v in x_values]
    else:
        x_values = df_sorted['parts_per_layer']
        x_label = 'Partitions per Layer'
        x_tick_labels = df_sorted['parts_per_layer'].astype(int)

    # ------------------------------------------------------------------ #
    # 9.  Plot setup
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    ax2 = ax1.twinx()  # secondary axis for error

    x_pos = np.arange(len(df_sorted))

    # ------------------------------------------------------------------ #
    # 10.  Bars (same as original)
    # ------------------------------------------------------------------ #
    outer_w = 0.85
    inner_w = outer_w * 0.50
    left_off  = -inner_w / 2.0
    right_off = +inner_w / 2.0

    col_comp = "#F76D55"      # red
    col_comm = "#4A83EC"      # blue

    # Computation bar
    comp = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
    ax1.bar([p + left_off  for p in x_pos], comp,
            width=inner_w, color=col_comp, edgecolor='black', linewidth=0.5,
            label='Comp PE', zorder=3)

    # Data-flow bar
    ax1.bar([p + right_off for p in x_pos], df_sorted['data_flow_cycles'],
            width=inner_w, color=col_comm, edgecolor='black', linewidth=0.5,
            label='Comm Flow', zorder=3)

    # Outer outline bar
    ax1.bar(x_pos, df_sorted['overall_latency_cycles'],
            width=outer_w, color='none', edgecolor='black', linewidth=0.65,
            hatch='///', label='All', zorder=2)

    # ------------------------------------------------------------------ #
    # 11. Scatter: Error (%) from csv_path2
    # ------------------------------------------------------------------ #
    flops_color = "#26073B"  # green
    size_color = "#600D13"   # teal
    
    if x_sizes:
        ax2.plot(x_pos, df_sorted['avg_flops'],
                    color=flops_color, alpha=0.95, marker='o', linewidth=0.75,
                    label='FLOPs', zorder=4)
        ax2.set_ylabel('FLOPs', fontweight='normal')
    elif x_flops:
        ax2.plot(x_pos, df_sorted['avg_size'] / 1024,  # Convert to KB
                    color=size_color, alpha=0.95, marker='o', linewidth=0.75,
                    label='Size (KB)', zorder=4)
        ax2.set_ylabel('Size (KB)', fontweight='normal')

    # ------------------------------------------------------------------ #
    # 12. Labels, ticks, grid
    # ------------------------------------------------------------------ #
    ax1.set_xlabel(x_label, fontweight='normal')
    ax1.set_ylabel('Latency (cycles)', fontweight='normal')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_tick_labels, rotation=45)

    # Set y-axis limits
    if y_max is not None:
        ax1.set_ylim(0, y_max)

    # Scientific notation for y-axis
    if y_scientific:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax1.yaxis.set_major_formatter(formatter)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(top=False, right=False)

    # ------------------------------------------------------------------ #
    # 13. Legend
    # ------------------------------------------------------------------ #
    if show_legend:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(h1 + h2, l1 + l2,
                loc='upper center', ncol=4, frameon=True, framealpha=1.0,
                edgecolor='black', fancybox=False, facecolor='white',
                bbox_to_anchor=(0.5, 1.05))
        # Ensure legend is drawn on top and not clipped
        legend.set_zorder(100)

    # Use tight_layout with rect to leave space for legend
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved → {save_path}")

    return fig, ax1, ax2

def plot_latency_breakdown_with_error_from_two_files_with_broken_axis(
        csv_path1,                     # latency breakdown file
        csv_path2,                     # error / comparison file
        figsize=(5.0, 2.5),
        save_path=None,
        dpi=300,
        min_parts_per_layer=None,
        max_parts_per_layer=None,
        take_avg=False,
        break_point=100.0):
    """
    IEEE-style plot:
      - Left y-axis (latency) is **continuous**
      - Right y-axis (error %) is **broken** at `break_point`
    """
    # ------------------------------------------------------------------ #
    # 1.  Style (Helvetica + IEEE)
    # ------------------------------------------------------------------ #
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # ------------------------------------------------------------------ #
    # 2.  Load & process CSVs
    # ------------------------------------------------------------------ #
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    if take_avg:
        num_cols = df1.select_dtypes(include='number').columns.tolist()
        agg_cols = [c for c in num_cols if c != 'parts_per_layer']
        agg_dict = {c: 'mean' for c in agg_cols}
        df1_grp = df1.groupby('parts_per_layer').agg(agg_dict).reset_index()
    else:
        df1_grp = df1.loc[df1.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]

    df2_grp = df2.groupby('parts_per_layer').agg({'percentage_diff': 'mean'}).reset_index()

    df_merged = pd.merge(df1_grp, df2_grp, on='parts_per_layer', how='inner')
    if df_merged.empty:
        raise ValueError("No overlapping 'parts_per_layer' values!")

    if min_parts_per_layer is not None:
        df_merged = df_merged[df_merged['parts_per_layer'] >= min_parts_per_layer]
    if max_parts_per_layer is not None:
        df_merged = df_merged[df_merged['parts_per_layer'] <= max_parts_per_layer]

    df_sorted = df_merged.sort_values('parts_per_layer').reset_index(drop=True)
    df_sorted['data_flow_percentage'] = (
        df_sorted['data_flow_cycles'] / df_sorted['overall_latency_cycles'] * 100
    )

    # ------------------------------------------------------------------ #
    # 3.  Figure with GridSpec: 2 rows for right axis, 1 shared left axis
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.05)

    # Left: continuous latency axis (spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])

    # Right: broken error axis
    ax2_upper = fig.add_subplot(gs[0, 1])  # error > break_point
    ax2_lower = fig.add_subplot(gs[1, 1])  # error <= break_point

    # ------------------------------------------------------------------ #
    # 4.  Bars on ax1 (continuous latency)
    # ------------------------------------------------------------------ #
    x_pos = np.arange(len(df_sorted))
    outer_w = 0.85
    inner_w = outer_w * 0.50
    left_off  = -inner_w / 2.0
    right_off = +inner_w / 2.0

    col_comp = "#F76D55"
    col_comm = "#4A83EC"

    comp = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
    ax1.bar([p + left_off  for p in x_pos], comp,
            width=inner_w, color=col_comp, edgecolor='black', linewidth=0.5,
            label='Computation', zorder=3)
    ax1.bar([p + right_off for p in x_pos], df_sorted['data_flow_cycles'],
            width=inner_w, color=col_comm, edgecolor='black', linewidth=0.5,
            label='Data Flow', zorder=3)
    ax1.bar(x_pos, df_sorted['overall_latency_cycles'],
            width=outer_w, color='none', edgecolor='black', linewidth=0.65,
            hatch='///', label='Comp + Comm', zorder=2)

    # ------------------------------------------------------------------ #
    # 5.  Scatter – lower part (<= break_point)
    # ------------------------------------------------------------------ #
    err_color = '#2ca02c'
    mask_lower = df_sorted['percentage_diff'] <= break_point
    if mask_lower.any():
        ax2_lower.scatter(x_pos[mask_lower], df_sorted['percentage_diff'][mask_lower],
                          color=err_color, marker='^', s=40, edgecolor='black',
                          linewidth=0.5, label='Error (%)', zorder=4)

    # ------------------------------------------------------------------ #
    # 6.  Scatter – upper part (> break_point)
    # ------------------------------------------------------------------ #
    mask_upper = df_sorted['percentage_diff'] > break_point
    if mask_upper.any():
        # Plot actual values (not offset) on upper axis
        ax2_upper.scatter(x_pos[mask_upper], df_sorted['percentage_diff'][mask_upper],
                          color=err_color, marker='^', s=40, edgecolor='black',
                          linewidth=0.5, zorder=4)

    # ------------------------------------------------------------------ #
    # 7.  Axis limits and alignment
    # ------------------------------------------------------------------ #
    ax1.set_ylim(0, df_sorted['overall_latency_cycles'].max() * 1.1)

    # Set x-limits to match across all axes
    x_min, x_max = -0.5, len(df_sorted) - 0.5
    ax1.set_xlim(x_min, x_max)
    ax2_lower.set_xlim(x_min, x_max)
    ax2_upper.set_xlim(x_min, x_max)

    # Y-limits for broken axis
    ax2_lower.set_ylim(0, break_point)

    if mask_upper.any():
        max_up = df_sorted['percentage_diff'].max()
        # Upper axis shows values from break_point to max
        ax2_upper.set_ylim(break_point, max_up * 1.05)
    else:
        ax2_upper.set_visible(False)

    # ------------------------------------------------------------------ #
    # 8.  Labels
    # ------------------------------------------------------------------ #
    ax1.set_xlabel('Partitions per Layer', fontweight='normal')
    ax1.set_ylabel('Latency (cycles)', fontweight='normal')
    ax2_lower.set_ylabel('Error (%)', fontweight='normal', labelpad=10)
    if mask_upper.any():
        ax2_upper.set_ylabel('Error (%)', fontweight='normal', labelpad=10)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)

    # ------------------------------------------------------------------ #
    # 9.  Grid (only on latency axis)
    # ------------------------------------------------------------------ #
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------ #
    # 10. Broken axis cut-out lines (only on right y-axis)
    # ------------------------------------------------------------------ #
    if mask_upper.any():
        d = 0.5
        kwargs = dict(
            marker=[(-1, -d), (1, d)], markersize=12,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False
        )
        ax2_upper.plot([0, 1], [0, 0], transform=ax2_upper.transAxes, **kwargs)
        ax2_lower.plot([0, 1], [1, 1], transform=ax2_lower.transAxes, **kwargs)

        # Hide spines between upper and lower error axes
        ax2_upper.spines['bottom'].set_visible(False)
        ax2_lower.spines['top'].set_visible(False)

    # Tick settings for right axes
    ax2_upper.xaxis.tick_top()
    ax2_upper.tick_params(labeltop=False, top=False, bottom=False)
    ax2_lower.xaxis.tick_bottom()
    ax2_lower.tick_params(labelbottom=False, top=False, bottom=False)

    # Hide x-tick labels on right axes
    plt.setp(ax2_upper.get_xticklabels(), visible=False)
    plt.setp(ax2_lower.get_xticklabels(), visible=False)

    # ------------------------------------------------------------------ #
    # 11. Spines & visibility
    # ------------------------------------------------------------------ #
    for ax in (ax1, ax2_lower, ax2_upper):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    if not mask_upper.any():
        ax2_upper.set_visible(False)

    # ------------------------------------------------------------------ #
    # 12. Legend (on left axis)
    # ------------------------------------------------------------------ #
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2_lower.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2,
               loc='upper left', frameon=True, framealpha=1.0,
               edgecolor='black', fancybox=False, facecolor='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")

    return fig, ax1, (ax2_lower, ax2_upper)
