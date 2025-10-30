import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

###############################################################################################
####                            NOC comparison functions                                    ###
###############################################################################################

def plot_noc_comparison_ieee(csv_path, figsize=(3.5, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with NoC comparison results
    figsize : tuple
        Figure size in inches (default: 3.5x2.5 for IEEE 2-column)
    save_path : str, optional
        Path to save the figure (if None, only displays)
    dpi : int
        Resolution for saved figure (default: 300)
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include (filters out smaller values)
    max_parts_per_layer : int, optional
        Maximum number of partitions per layer to include (filters out larger values)
    
    Returns:
    --------
    fig, ax1, ax2 : matplotlib objects
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
        'lines.markersize': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # For duplicate parts_per_layer, average the percentage_diff and time_gain
    df_averaged = df.groupby('parts_per_layer').agg({
        'percentage_diff': 'mean',
        'time_gain': 'mean',
        'num_partitions': 'first',  # Keep first value (should be same for duplicates)
        'parts_per_layer': 'first'
    }).reset_index(drop=True)
    
    # Apply min/max partitions filter if specified
    if min_parts_per_layer is not None:
        df_averaged = df_averaged[df_averaged['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_averaged = df_averaged[df_averaged['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer for better visualization
    df_sorted = df_averaged.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After averaging duplicates and filtering: {len(df_sorted)} rows")
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    
    # Plot error (percentage_diff) on left y-axis
    color1 = '#1f77b4'  # Blue
    ax1.scatter(df_sorted['parts_per_layer'], df_sorted['percentage_diff'], 
                color=color1, marker='o', s=25, alpha=0.7, label='Error (%)', zorder=3)
    ax1.set_xlabel('Partitions per Layer', fontweight='normal')
    ax1.set_ylabel('Error (%)', fontweight='normal')  # No color
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(axis='x', direction='in')
    
    # Add horizontal grid
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Create second y-axis for speedup with LOG SCALE
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'  # Orange
    ax2.scatter(df_sorted['parts_per_layer'], df_sorted['time_gain'], 
                color=color2, marker='s', s=25, alpha=0.7, label='Speedup', zorder=3)
    ax2.set_ylabel('Speedup (×)', fontweight='normal')  # No color
    ax2.set_yscale('log')  # LOG SCALE for speedup
    ax2.tick_params(axis='y', direction='in')
    
    # Make top and right spines visible without ticks
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    
    # Remove ticks from top axis
    ax1.tick_params(top=False, labeltop=False)
    ax2.tick_params(top=False, labeltop=False)
    
    # Add combined legend - FULLY OPAQUE
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='best', frameon=True, framealpha=1.0, edgecolor='black', 
              fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax1, ax2


def plot_noc_comparison_latencies_ieee(csv_path, figsize=(3.5, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file with NoC comparison results
    figsize : tuple
        Figure size in inches (default: 3.5x2.5 for IEEE 2-column)
    save_path : str, optional
        Path to save the figure (if None, only displays)
    dpi : int
        Resolution for saved figure (default: 300)
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    max_parts_per_layer : int, optional
        Maximum number of partitions per layer to include (filters out larger values)

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
        'lines.markersize': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # Read data
    df = pd.read_csv(csv_path)

    # For duplicate parts_per_layer, average the result_booksim and result_analytical
    df_averaged = df.groupby('parts_per_layer').agg({
        'result_booksim': 'mean',
        'result_analytical': 'mean',
        'num_partitions': 'first',  # Keep first value (should be same for duplicates)
        'parts_per_layer': 'first'
    }).reset_index(drop=True)

    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_averaged = df_averaged[df_averaged['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_averaged = df_averaged[df_averaged['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")

    # Sort by parts_per_layer for better visualization
    df_sorted = df_averaged.sort_values('parts_per_layer')

    print(f"Original data: {len(df)} rows")
    print(f"After averaging duplicates and filtering: {len(df_sorted)} rows")

    # Create figure with single axis
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Colors
    color1 = "#339feb"  # Blue
    color2 = "#f3933f"  # Orange

    # Plot Booksim latency with scatter and line
    ax.plot(df_sorted['parts_per_layer'], df_sorted['result_booksim'],
            color=color1, marker='o', markersize=4, linewidth=1.5,
            label='Booksim', zorder=3)

    # Plot Lightweight latency with scatter and line
    ax.plot(df_sorted['parts_per_layer'], df_sorted['result_analytical'],
            color=color2, marker='s', markersize=4, linewidth=1.5,
            label='Lightweight', zorder=3)

    # Set labels
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Latency (cycles)', fontweight='normal')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')

    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Make top and right spines visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Remove ticks from top and right
    ax.tick_params(top=False, right=False)

    # Add legend - FULLY OPAQUE
    ax.legend(loc='best', frameon=True, framealpha=1.0, edgecolor='black',
              fancybox=False, facecolor='white')

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig, ax


###############################################################################################
####                            Latencies breakdowns                                        ###
###############################################################################################

def plot_latency_breakdown_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=False, max_parts_per_layer=False, take_avg=False):
    """    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    
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
    
    if take_avg:
        # Average duplicate parts_per_layer entries - specify which columns to average
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Remove parts_per_layer from aggregation since we're grouping by it
        agg_cols = [col for col in numeric_cols if col != 'parts_per_layer']
        
        # Create aggregation dict: mean for numeric columns
        agg_dict = {col: 'mean' for col in agg_cols}
        
        df_filtered = df.groupby('parts_per_layer').agg(agg_dict).reset_index()
    else:
        # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
        df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Calculate data flow percentage
    df_sorted = df_sorted.copy()
    df_sorted['data_flow_percentage'] = (df_sorted['data_flow_cycles'] / 
                                         df_sorted['overall_latency_cycles'] * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    x_positions = range(len(df_sorted))
    bar_width = 0.8
    
    # Colors
    color_overall = "#F76D55"  # Red for overall latency
    color_dataflow =  "#4A83EC" # Blue for data flow overlay
    
    # Plot overall latency as base bars
    bars1 = ax.bar(x_positions, df_sorted['overall_latency_cycles'], 
                   width=bar_width,
                   color=color_overall, label='Overall Latency', 
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    # Overlay data flow cycles on top (starting from bottom)
    bars2 = ax.bar(x_positions, df_sorted['data_flow_cycles'], 
                   width=bar_width,
                   color=color_dataflow, label='Data Flow', 
                   edgecolor='black', linewidth=0.5, zorder=3,
                   alpha=0.85)  # Slight transparency to show it's an overlay
    
    # Set labels and formatting
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Latency (cycles)', fontweight='normal')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)
    
    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Ticks inside
    ax.tick_params(axis='both', direction='in')
    
    # Make top spine visible without ticks
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    
    # Add legend - fully opaque
    ax.legend(loc='best', frameon=True, framealpha=1.0, 
             edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nLatency Statistics:")
    print(f"  Min overall latency: {df_sorted['overall_latency_cycles'].min():.0f} cycles")
    print(f"  Max overall latency: {df_sorted['overall_latency_cycles'].max():.0f} cycles")
    print(f"  Min data flow %: {df_sorted['data_flow_percentage'].min():.1f}%")
    print(f"  Max data flow %: {df_sorted['data_flow_percentage'].max():.1f}%")
    print(f"  Mean data flow %: {df_sorted['data_flow_percentage'].mean():.1f}%")
    print(f"\nData flow percentage trend (showing increasing communication overhead):")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: {row['data_flow_percentage']:.1f}% data flow")
    
    return fig, ax



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
        'xtick.labelsize': 7,
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
    #old options here: Computations only part (no overlapping data flow)
    
    comp_cycles = df_sorted['comp_cycles']
    
    comp_cycles_only = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
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
            label='Data Flow',
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



def plot_latency_breakdown_4_bars(
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
        'xtick.labelsize': 7,
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
    inner_width = outer_width / 3.0  # Divide by 3 for three bars
    left_offset  = -inner_width
    middle_offset = 0.0
    right_offset = +inner_width

    # Colors
    color_comp_total = "#F76D55"   # red - total computation
    color_comp_pure = "#F48422"    # orange - pure computation (no overlap)
    color_dataflow = "#4A83EC"     # blue - data flow

    # Calculate cycles
    comp_cycles = df_sorted['comp_cycles']
    comp_cycles_only_2= comp_cycles - df_sorted['overlapping_cycles']
    
    comp_cycles_only = df_sorted['overall_latency_cycles'] - df_sorted['data_flow_cycles']
    
    dataflow_only = df_sorted['data_flow_cycles'] - df_sorted['overlapping_cycles']
    
    comp_cycles_only3 = comp_cycles - df_sorted['overlapping_cycles'] - df_sorted['idle_cycles']

    ### 4 + 1 bars: Computation total, data flow, overlapping, idle
    breakpoint()
    
    # 1. Left inner bar – Total Computation
    ax.bar([p + left_offset for p in x_positions],
            comp_cycles,
            width=inner_width,
            color=color_comp_total,
            edgecolor='black',
            linewidth=0.5,
            label='Comp PE',
            zorder=3)

    # 2. Middle inner bar – Pure Computation (no overlap)
    ax.bar([p + middle_offset for p in x_positions],
            comp_cycles_only,
            width=inner_width,
            color=color_comp_pure,
            edgecolor='black',
            linewidth=0.5,
            label='Comp PE (pure)',
            zorder=3)

    # 3. Right inner bar – Data Flow
    ax.bar([p + right_offset for p in x_positions],
            df_sorted['data_flow_cycles'],
            width=inner_width,
            color=color_dataflow,
            edgecolor='black',
            linewidth=0.5,
            label='Data Flow',
            zorder=3)

    # 4. Outer hatched bar – Overall
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

    # Legend (4 items: Comp PE, Comp PE (pure), Data Flow, All)
    if show_legend:
        ax.legend(loc='upper center', ncol=4, frameon=True, framealpha=1.0,
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
        y_max=None,
        max_error=None):
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
    max_error : float, optional
        Set maximum value for error y-axis (y2). If None, uses automatic scaling

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
        'xtick.labelsize': 7,
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
            label='Data Flow', zorder=3)

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

    # Set error y-axis limits
    if max_error is not None:
        ax2.set_ylim(0, max_error)

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
        'xtick.labelsize': 7,
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
            label='Data Flow', zorder=3)

    # Outer outline bar
    ax1.bar(x_pos, df_sorted['overall_latency_cycles'],
            width=outer_w, color='none', edgecolor='black', linewidth=0.65,
            hatch='///', label='All', zorder=2)

    # ------------------------------------------------------------------ #
    # 11. Scatter: Error (%) from csv_path2
    # ------------------------------------------------------------------ #
    flops_color = "#28073E"  # green
    size_color = "#44090D"   # teal
    
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
        'xtick.labelsize': 7,
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


###############################################################################################
####                            Energies breakdowns                                        ###
###############################################################################################



def plot_energy_breakdown_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """
    Create IEEE conference paper style stacked bar plot for energy breakdown.
    
    Shows total energy as stacked bars: PE energy + Data flow energy.
    Note: energy_PEs_uJ + energy_data_flow_uJ = total_energy_uJ
    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    
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
    
    # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
    df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Calculate data flow energy percentage
    df_sorted = df_sorted.copy()
    df_sorted['data_flow_energy_percentage'] = (df_sorted['energy_data_flow_uJ'] / 
                                                 df_sorted['total_energy_uJ'] * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    x_positions = range(len(df_sorted))
    bar_width = 0.8
    
    # Colors - different from latency plot
    color_pe = "#46EA8A"  # Green for PE energy
    color_dataflow = '#9B59B6'  # Purple for data flow energy
    
    # Plot stacked bars
    # Bottom part: PE energy
    bars1 = ax.bar(x_positions, df_sorted['energy_PEs_uJ'], 
                   width=bar_width,
                   color=color_pe, label='PE Energy', 
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    # Top part: Data flow energy
    bars2 = ax.bar(x_positions, df_sorted['energy_data_flow_uJ'], 
                   bottom=df_sorted['energy_PEs_uJ'],
                   width=bar_width,
                   color=color_dataflow, label='Data Flow Energy', 
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    # Set labels and formatting
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Energy (µJ)', fontweight='normal')  # microjoules symbol
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)
    
    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Ticks inside
    ax.tick_params(axis='both', direction='in')
    
    # Make top spine visible without ticks
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    
    # Add legend - fully opaque
    ax.legend(loc='best', frameon=True, framealpha=1.0, 
             edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nEnergy Statistics:")
    print(f"  Min total energy: {df_sorted['total_energy_uJ'].min():.2f} µJ")
    print(f"  Max total energy: {df_sorted['total_energy_uJ'].max():.2f} µJ")
    print(f"  Min data flow energy %: {df_sorted['data_flow_energy_percentage'].min():.1f}%")
    print(f"  Max data flow energy %: {df_sorted['data_flow_energy_percentage'].max():.1f}%")
    print(f"  Mean data flow energy %: {df_sorted['data_flow_energy_percentage'].mean():.1f}%")
    print(f"\nData flow energy percentage trend:")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: {row['data_flow_energy_percentage']:.1f}% data flow energy")
    
    return fig, ax


#############################################################################################################
####                            Max / Avg packet delay breakdowns                                       #####
############################################################################################################


def plot_max_packet_delay_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """
    Create IEEE conference paper style bar plot for maximum packet delay.
    
    Shows how maximum packet delay changes with partitioning strategy.
    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    
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
    
    # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
    df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    x_positions = range(len(df_sorted))
    bar_width = 0.8
    
    # Color - using orange for packet delay
    color_delay = "#F49A4B"  # Orange for max packet delay
    
    # Plot bars
    bars = ax.bar(x_positions, df_sorted['max_delay_packets_cycles'], 
                  width=bar_width,
                  color=color_delay, label='Max Packet Delay', 
                  edgecolor='black', linewidth=0.5, zorder=2)
    
    # Set labels and formatting
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Max Packet Delay (cycles)', fontweight='normal')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)
    
    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Ticks inside
    ax.tick_params(axis='both', direction='in')
    
    # Make top spine visible without ticks
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    
    # Add legend - fully opaque
    #ax.legend(loc='best', frameon=True, framealpha=1.0, 
    #         edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nMax Packet Delay Statistics:")
    print(f"  Min max delay: {df_sorted['max_delay_packets_cycles'].min():.0f} cycles")
    print(f"  Max max delay: {df_sorted['max_delay_packets_cycles'].max():.0f} cycles")
    print(f"  Mean max delay: {df_sorted['max_delay_packets_cycles'].mean():.0f} cycles")
    print(f"\nMax packet delay trend:")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: {row['max_delay_packets_cycles']:.0f} cycles")
    
    return fig, ax


def plot_avg_packet_size_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """
    Create IEEE conference paper style bar plot for maximum packet delay.
    
    Shows how maximum packet delay changes with partitioning strategy.
    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    
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
    
    # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
    df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    x_positions = range(len(df_sorted))
    bar_width = 0.8
    
    # Color - using orange for packet delay
    color_delay = "#6050EE"  # Orange for max packet delay
    
    # Plot bars
    bars = ax.bar(x_positions, df_sorted['avg_packet_size_bytes'], 
                  width=bar_width,
                  color=color_delay, label='Avg Packet Size', 
                  edgecolor='black', linewidth=0.5, zorder=2)
    
    # Set labels and formatting
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Avg Packet Size (B)', fontweight='normal')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)
    
    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Ticks inside
    ax.tick_params(axis='both', direction='in')
    
    # Make top spine visible without ticks
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    
    # Add legend - fully opaque
    #ax.legend(loc='best', frameon=True, framealpha=1.0, 
    #         edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nAvg Packet Size Statistics:")
    print(f"  Min max delay: {df_sorted['avg_packet_size_bytes'].min():.0f} Bytes")
    print(f"  Max max delay: {df_sorted['avg_packet_size_bytes'].max():.0f} Bytes")
    print(f"  Mean max delay: {df_sorted['avg_packet_size_bytes'].mean():.0f} Bytes")
    print(f"\nAvg packet size trend:")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: {row['avg_packet_size_bytes']:.0f} bytes")
    
    return fig, ax



def plot_packet_delay_and_size_combined_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None):
    """
    Create IEEE conference paper style grouped bar plot combining max packet delay and avg packet size.
    
    Shows max packet delay and average packet size side-by-side with dual y-axes.
    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    
    Returns:
    --------
    fig, ax1, ax2 : matplotlib objects
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
    
    # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
    df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    
    # Set up grouped bars
    x_positions = np.arange(len(df_sorted))
    bar_width = 0.35
    
    # Colors for the two metrics
    color_delay = "#F49A4B"  # Orange for max packet delay
    color_size = "#6050EE"   # Purple for avg packet size
    
    # Plot max packet delay on left y-axis (Y1)
    bars1 = ax1.bar(x_positions - bar_width/2, df_sorted['max_delay_packets_cycles'].values, 
                    width=bar_width,
                    color=color_delay, label='Max Packet Delay', 
                    edgecolor='black', linewidth=0.5, zorder=2)
    
    ax1.set_xlabel('Partitions per Layer', fontweight='normal')
    ax1.set_ylabel('Max Packet Delay (cycles)', fontweight='normal')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(df_sorted['parts_per_layer'].astype(int).values, rotation=45)
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(axis='x', direction='in')
    
    # Add horizontal grid
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)
    
    # Create second y-axis for avg packet size (Y2)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_positions + bar_width/2, df_sorted['avg_packet_size_bytes'].values, 
                    width=bar_width,
                    color=color_size, label='Average Packet Size', 
                    edgecolor='black', linewidth=0.5, zorder=2)
    
    ax2.set_ylabel('Average Packet Size (B)', fontweight='normal')
    ax2.tick_params(axis='y', direction='in')
    
    # Make top spine visible without ticks
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    
    # Remove ticks from top axis
    ax1.tick_params(top=False, labeltop=False)
    ax2.tick_params(top=False, labeltop=False)
    
    # Add combined legend - fully opaque
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='best', frameon=True, framealpha=1.0, 
              edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nMax Packet Delay Statistics:")
    print(f"  Min: {df_sorted['max_delay_packets_cycles'].min():.0f} cycles")
    print(f"  Max: {df_sorted['max_delay_packets_cycles'].max():.0f} cycles")
    print(f"  Mean: {df_sorted['max_delay_packets_cycles'].mean():.0f} cycles")
    
    print(f"\nAverage Packet Size Statistics:")
    print(f"  Min: {df_sorted['avg_packet_size_bytes'].min():.0f} B")
    print(f"  Max: {df_sorted['avg_packet_size_bytes'].max():.0f} B")
    print(f"  Mean: {df_sorted['avg_packet_size_bytes'].mean():.0f} B")
    
    print(f"\nCombined trend:")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: "
              f"delay={row['max_delay_packets_cycles']:.0f} cycles, "
              f"size={row['avg_packet_size_bytes']:.0f} B")
    
    return fig, ax1, ax2


def plot_parallel_computations_ieee(csv_path, figsize=(5.0, 2.5), save_path=None, dpi=300, min_parts_per_layer=None, max_parts_per_layer=None, max_parallel_ref=None):
    """
    Create IEEE conference paper style grouped bar plot for parallel computations.
    
    Shows max and average parallel computations as percentages side-by-side for each partition level.
    
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
    min_parts_per_layer : int, optional
        Minimum number of partitions per layer to include in plot (filters out smaller values)
    max_parallel_ref : int, optional
        Reference value for 100% parallelism (e.g., 144 PEs = 100%)
        If None, uses the maximum value found in data
    
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
    
    # For duplicate parts_per_layer, keep only the one with smallest overall_latency_cycles
    df_filtered = df.loc[df.groupby('parts_per_layer')['overall_latency_cycles'].idxmin()]
    
    # Apply minimum partitions filter if specified
    if min_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] >= min_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer >= {min_parts_per_layer}")
    if max_parts_per_layer is not None:
        df_filtered = df_filtered[df_filtered['parts_per_layer'] <= max_parts_per_layer]
        print(f"Filtering: showing only parts_per_layer <= {max_parts_per_layer}")
    
    # Sort by parts_per_layer
    df_sorted = df_filtered.sort_values('parts_per_layer')
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering duplicates and applying min filter: {len(df_sorted)} rows")
    
    # Determine reference value for 100%
    if max_parallel_ref is None:
        max_parallel_ref = df_sorted['max_parallel_computations'].max()
        print(f"Using max value in data as 100% reference: {max_parallel_ref:.1f}")
    else:
        print(f"Using user-specified 100% reference: {max_parallel_ref}")
    
    # Convert to percentages
    df_sorted = df_sorted.copy()
    df_sorted['max_parallel_pct'] = (df_sorted['max_parallel_computations'] / max_parallel_ref) * 100
    df_sorted['avg_parallel_pct'] = (df_sorted['avg_parallel_computations'] / max_parallel_ref) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Set up grouped bars
    x_positions = np.arange(len(df_sorted))
    bar_width = 0.35
    
    # Colors for the two bars
    color_max = '#3498DB'  # Blue for max parallel
    color_avg = '#E74C3C'  # Red for avg parallel
    
    # Plot grouped bars with percentages
    bars1 = ax.bar(x_positions - bar_width/2, df_sorted['max_parallel_pct'], 
                   width=bar_width,
                   color=color_max, label='Max', 
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    bars2 = ax.bar(x_positions + bar_width/2, df_sorted['avg_parallel_pct'], 
                   width=bar_width,
                   color=color_avg, label='Average', 
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    # Set labels and formatting
    ax.set_xlabel('Partitions per Layer', fontweight='normal')
    ax.set_ylabel('Parallel Computations (%)', fontweight='normal')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['parts_per_layer'].astype(int), rotation=45)
    
    # Add horizontal grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Ticks inside
    ax.tick_params(axis='both', direction='in')
    
    # Make top spine visible without ticks
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    
    # Add legend - fully opaque
    ax.legend(loc='best', frameon=True, framealpha=1.0, 
              edgecolor='black', fancybox=False, facecolor='white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Print statistics
    print(f"\nParallel Computations Statistics (100% = {max_parallel_ref}):")
    print(f"  Max parallel - range: {df_sorted['max_parallel_pct'].min():.1f}% to {df_sorted['max_parallel_pct'].max():.1f}%")
    print(f"  Avg parallel - range: {df_sorted['avg_parallel_pct'].min():.1f}% to {df_sorted['avg_parallel_pct'].max():.1f}%")
    print(f"\nParallel computations trend:")
    for idx, row in df_sorted.iterrows():
        print(f"  {int(row['parts_per_layer'])} parts/layer: max={row['max_parallel_pct']:.1f}% ({row['max_parallel_computations']:.0f}), avg={row['avg_parallel_pct']:.1f}% ({row['avg_parallel_computations']:.1f})")
    
    return fig, ax