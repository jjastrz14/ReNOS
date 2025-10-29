import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
