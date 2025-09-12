'''
==================================================
File: ani_utils.py
Project: utils
File Created: Thursday, 23rd January 2025
Author: Jakub Jastrzebski, Edoardo Cabiati (jakubandrze.jastrzebski@polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The ani_utils.py module contains the functions used to generate the animation of the activity on the NoC grid.
The general framework to create the animation as been taken from https://github.com/jmjos/ratatoskr/blob/master/bin/plot_network.py
"""

import matplotlib.pyplot as plt
from typing import Union
import sys
import os
import numpy as np
import json
import seaborn as sns
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

import nocsim

'''#backup: 
GLOBAL_EVENT_COLORS = {
    'comp': ('#FF6347', 0.9),           # tomato - Computation
    'recon': ('#2E8B57', 0.9),          # seagreen - Reconfiguration  
    'traf_out': ('#6495ED', 0.7),       # cornflowerblue - Traffic PE out
    'traf_in': ('#FFA500', 0.7),        # orange - Traffic PE in
    'traf_between': ('#C0C0C0', 0.3),   # silver - Traffic NoC
    'reply_out': ('#FFD700', 0.5),      # gold - Reply PE out
    'reply_in': ('#40E0D0', 0.7),       # turquoise - Reply PE in
    'process': ('#45B7D1', 0.8),        # existing process color
    'local_process': ('#98D8C8', 0.9),  # existing local process color
}'''

# Global color and alpha configuration for consistent plotting across all classes
GLOBAL_EVENT_COLORS = {
    'comp': ('#FF6347', 0.15),           # tomato - Computation
    'recon': ('#2E8B57', 0.0),          # seagreen - Reconfiguration  
    'traf_out': ('#6495ED', 0.7),       # cornflowerblue - Traffic PE out
    'traf_in': ('#FFA500', 0.1),        # orange - Traffic PE in
    'traf_between': ('#C0C0C0', 0.0),   # silver - Traffic NoC
    'reply_out': ('#FFD700', 0.0),      # gold - Reply PE out
    'reply_in': ('#40E0D0', 0.0),       # turquoise - Reply PE in
    'process': ('#45B7D1', 0.0),        # existing process color
    'local_process': ('#98D8C8', 0.0),  # existing local process color
}

def get_event_color_and_alpha(event_type):
    """Helper function to get color and alpha for an event type"""
    return GLOBAL_EVENT_COLORS.get(event_type, ('#95A5A6', 0.8))  # Default gray color


@dataclass
class NodeEvent:
    """Represents an event on a node"""
    node_id: int
    task_id: int
    task_type: str
    start_time: int
    end_time: int
    event_type: str  # 'send', 'compute', 'receive', 'process'
    related_node: int = None  # for network operations
    size: int = 0
    description: str = ""

class NoCSimulationVisualizer:
    """Visualization tool for NoC simulation"""
    
    def __init__(self):
        self.events = []
        self.node_states = defaultdict(list)
        self.total_nodes = 0
        self.max_time = 0
        
    def simulate_with_tracking(self, simulator, workload, arch):
        """Enhanced simulation that tracks all node events using the real simulator"""
        
        # Track all nodes mentioned in workload
        all_nodes = set()
        for task in workload:
            if hasattr(task, 'src') and task.src is not None:
                all_nodes.add(task.src)
            if hasattr(task, 'dst') and task.dst is not None:
                all_nodes.add(task.dst)
            if hasattr(task, 'node') and task.node is not None:
                all_nodes.add(task.node)
        
        self.total_nodes = max(all_nodes) + 1 if all_nodes else 0
        
        # Use the real simulator to get accurate results including batching optimizations
        total_latency = simulator.simulate_execution(workload, arch, self)
        self.max_time = total_latency
        
        return total_latency
    
    # Implement EventTracker interface methods
    def track_comp_op_start(self, task_id: int, node: int, start_time: int, ct_required: int) -> None:
        """Track the start of a COMP_OP operation"""
        #print(f"DEBUG: track_comp_op_start - task_id={task_id}, node={node}, start_time={start_time}, ct_required={ct_required}")
        self.add_event(NodeEvent(
            node_id=node,
            task_id=task_id,
            task_type='COMP_OP',
            start_time=start_time,
            end_time=start_time,  # Will be updated by track_comp_op_end
            event_type='comp_start',
            description=f"Start COMP_OP task {task_id} (cycles: {ct_required})"
        ))
    
    def track_comp_op_end(self, task_id: int, node: int, end_time: int) -> None:
        """Track the completion of a COMP_OP operation"""
        #print(f"DEBUG: track_comp_op_end - task_id={task_id}, node={node}, end_time={end_time}")
        # Find and update the existing start event
        updated = False
        for event in self.events:
            if event.task_id == task_id and event.node_id == node and event.event_type == 'comp_start':
                event.end_time = end_time
                event.event_type = 'comp'
                event.description = f"COMP_OP task {task_id} completed"
                updated = True
                break
        if not updated:
            print(f"DEBUG: WARNING - Could not find compute_start event for task {task_id}")
        #print(f"DEBUG: Total events after comp_op_end: {len(self.events)}")
        
    def track_write_send_start(self, task_id: int, src: int, dst: int, start_time: int, size: int) -> None:
        """Track the start of sending a WRITE message"""
        self.add_event(NodeEvent(
            node_id=src,
            task_id=task_id,
            task_type='WRITE',
            start_time=start_time,
            end_time=start_time,  # Will be updated by track_write_send_end
            event_type='traf_out_start',
            related_node=dst,
            size=size,
            description=f"Start sending WRITE to node {dst} (size: {size})"
        ))
        
    def track_write_send_end(self, task_id: int, src: int, dst: int, end_time: int) -> None:
        """Track the end of sending a WRITE message"""
        # Find and update the existing send start event
        for event in self.events:
            if event.task_id == task_id and event.node_id == src and event.event_type == 'traf_out_start':
                event.end_time = end_time
                event.event_type = 'traf_out'
                event.description = f"Send WRITE to node {dst} completed"
                break
        
    def track_write_receive_start(self, task_id: int, src: int, dst: int, start_time: int) -> None:
        """Track the start of receiving a WRITE message"""
        self.add_event(NodeEvent(
            node_id=dst,
            task_id=task_id,
            task_type='WRITE',
            start_time=start_time,
            end_time=start_time,  # Will be updated when receive completes
            event_type='traf_in',
            related_node=src,
            description=f"Receive WRITE from node {src}"
        ))
        
    def track_write_process_start(self, task_id: int, dst: int, start_time: int, pt_required: int) -> None:
        """Track the start of processing a WRITE message"""
        self.add_event(NodeEvent(
            node_id=dst,
            task_id=task_id,
            task_type='WRITE',
            start_time=start_time,
            end_time=start_time,  # Will be updated by track_write_process_end
            event_type='process_start',
            description=f"Start processing WRITE task {task_id} (pt_required: {pt_required})"
        ))
        
    def track_write_process_end(self, task_id: int, dst: int, end_time: int) -> None:
        """Track the end of processing a WRITE message"""
        # Find and update the existing process start event
        for event in self.events:
            if event.task_id == task_id and event.node_id == dst and event.event_type == 'process_start':
                event.end_time = end_time
                event.event_type = 'process'
                event.description = f"Process WRITE task {task_id} completed"
                break
        
    def track_reply_send_start(self, task_id: int, src: int, dst: int, start_time: int) -> None:
        """Track the start of sending a reply"""
        self.add_event(NodeEvent(
            node_id=src,
            task_id=task_id,
            task_type='REPLY',
            start_time=start_time,
            end_time=start_time,  # Will be updated when reply completes
            event_type='reply_out_start',
            related_node=dst,
            description=f"Send REPLY to node {dst}"
        ))
        
    def track_reply_receive_end(self, task_id: int, src: int, dst: int, end_time: int) -> None:
        """Track the end of receiving a reply"""
        self.add_event(NodeEvent(
            node_id=dst,
            task_id=task_id,
            task_type='REPLY',
            start_time=end_time,  # Simplified - assume instantaneous receive
            end_time=end_time,
            event_type='reply_in',
            related_node=src,
            description=f"Receive REPLY from node {src}"
        ))
        
    def track_write_receive_end(self, task_id: int, src: int, dst: int, end_time: int) -> None:
        """Track the end of receiving and processing a write at destination"""
        self.add_event(NodeEvent(
            node_id=dst,
            task_id=task_id,
            task_type='WRITE',
            start_time=end_time-1,  # Simplified - assume processing took some time
            end_time=end_time,
            event_type='process',
            related_node=src,
            description=f"Process WRITE from node {src}"
        ))
        
    def track_reply_send_end(self, task_id: int, src: int, dst: int, end_time: int) -> None:
        """Track the end of sending a reply"""
        # Find and update the existing reply send event
        for event in self.events:
            if event.task_id == task_id and event.node_id == src and event.event_type == 'reply_out_start':
                event.end_time = end_time
                event.event_type = 'reply_out'
                break
        
    def track_batch_comp_ops(self, task_ids: List[int], node: int, start_time: int, end_time: int, layer_id: int) -> None:
        """Track a batch of COMP_OP operations executed together"""
        # Create events for the batched operations
        for task_id in task_ids:
            self.add_event(NodeEvent(
                node_id=node,
                task_id=task_id,
                task_type='COMP_OP',
                start_time=start_time,
                end_time=end_time,
                event_type='comp',
                description=f"Batched COMP_OP from layer {layer_id} (batch of {len(task_ids)} ops)"
            ))
    
    def track_batched_write(self, task_ids: List[int], src: int, dst: int, start_time: int, end_time: int, total_size: int) -> None:
        """Track a batched WRITE operation"""
        # Create a single event for the batched write
        self.add_event(NodeEvent(
            node_id=src,
            task_id=task_ids[0],  # Use first task ID as representative
            task_type='WRITE_BATCH',
            start_time=start_time,
            end_time=end_time,
            event_type='traf_out',
            related_node=dst,
            size=total_size,
            description=f"Batched WRITE to node {dst} (batch of {len(task_ids)} writes, total size: {total_size})"
        ))
    

    
    def add_event(self, event: NodeEvent):
        """Add an event to tracking"""
        self.events.append(event)
        self.node_states[event.node_id].append(event)
    
    def plot_timeline(self, figsize=(10, 6), save_path=None):
        """Create a Gantt chart showing node activities over time"""
        if not self.events:
            print("No events to plot. Run simulation first.")
            return
        
        # Debug: print event summary
        #print(f"DEBUG: Total events to plot: {len(self.events)}")
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        #print(f"DEBUG: Event types: {event_types}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_positions = {}
        current_y = 0
        
        # Sort events by node for cleaner visualization
        for node_id in sorted(self.node_states.keys()):
            y_positions[node_id] = current_y
            current_y += 1
        
        # Create legend labels mapping - define before use (match BookSim2 order)
        legend_labels = {
            'comp': 'Computation',
            'recon': 'Reconfiguration', 
            'traf_out': 'Traffic PE out',
            'traf_in': 'Traffic PE in',
            'traf_between': 'Traffic NoC',
            'reply_in': 'Reply PE in',
            'reply_out': 'Reply PE out',
            'process': 'Process Data',
            'local_process': 'Local Process'
        }
        
        # Define event types order to match BookSim2
        event_types = [
            (event_type, color, legend_labels[event_type], alpha)
            for event_type, (color, alpha) in GLOBAL_EVENT_COLORS.items()
            if event_type in legend_labels
        ]
        
        # Group events by node and event type like BookSim2
        node_event_data = {}
        for node_id in sorted(self.node_states.keys()):
            node_event_data[node_id] = {}
            for event in self.node_states[node_id]:
                event_type = event.event_type
                if event_type not in node_event_data[node_id]:
                    node_event_data[node_id][event_type] = []
                duration = event.end_time - event.start_time
                node_event_data[node_id][event_type].append((event.start_time, duration))
        
        # Track used labels to prevent duplicates in legend
        used_labels = set()
        
        # Plot events using broken_barh exactly like BookSim2
        for node_id, events in node_event_data.items():
            has_events = False  # Flag to check if node has any events
            
            for event_key, color, label, alpha in event_types:
                event_list = events.get(event_key, [])
                if event_list:
                    has_events = True
                    # Use label only if it hasn't been used before
                    current_label = label if label not in used_labels else None
                    ax.broken_barh(
                        event_list,
                        (node_id - 0.4, 0.8),
                        facecolors=color,
                        label=current_label,
                        alpha=alpha
                    )
                    if current_label:
                        used_labels.add(label)
            
            # Handle nodes with no events
            if not has_events:
                print(f"No events found for node {node_id}")
            
            # you can add text labels if the rectangle is wide enough
        
        # Set y-ticks to node IDs (match BookSim2)
        ax.set_yticks(range(len(y_positions)))
        ax.set_yticklabels([f'{node_id}' for node_id in sorted(y_positions.keys())])
        
        # Set x-ticks to cycle numbers (match BookSim2)
        ax.set_xlim(0, self.max_time)
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel('Node', fontsize=12)
        
        # Auto-adjust ticks like BookSim2
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        # Add vertical lines at each major x-tick (match BookSim2)
        for tick in ax.xaxis.get_major_locator().tick_values(ax.get_xlim()[0], ax.get_xlim()[1]):
            ax.axvline(x=tick, color='grey', linestyle='-', linewidth=0.5, zorder=0)

        # Add horizontal lines at the corners of the nodes (match BookSim2)
        for node in range(len(y_positions)):
            ax.axhline(y=node - 0.4, color='grey', linestyle='--', linewidth=0.5)
            ax.axhline(y=node + 0.4, color='grey', linestyle='--', linewidth=0.5)
        
        # Create legend elements in the same order as BookSim2
        legend_elements = []
        for event_key, color, label, alpha in event_types:
            if label in used_labels:
                legend_elements.append(patches.Patch(color=color, label=label))
        
        # Use BookSim2 style legend placement
        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper center',
                ncol=6,
                bbox_to_anchor=(0.5, 1.05),
                fancybox=True,
                shadow=True,
                title_fontsize='medium',
                frameon=True
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_node_utilization(self, figsize=(12, 8), save_path=None):
        """Plot utilization statistics for each node"""
        if not self.events:
            print("No events to plot. Run simulation first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate utilization for each node
        node_utilization = {}
        node_activity_breakdown = defaultdict(lambda: defaultdict(int))
        
        for node_id in range(self.total_nodes):
            total_busy_time = 0
            activity_times = defaultdict(int)
            
            for event in self.node_states[node_id]:
                duration = event.end_time - event.start_time
                total_busy_time += duration
                activity_times[event.event_type] += duration
            
            node_utilization[node_id] = (total_busy_time / self.max_time) * 100 if self.max_time > 0 else 0
            node_activity_breakdown[node_id] = activity_times
        
        # Plot 1: Overall utilization
        nodes = list(range(self.total_nodes))
        utilizations = [node_utilization.get(node, 0) for node in nodes]
        
        bars = ax1.bar(nodes, utilizations, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Node ID')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('Node Utilization')
        ax1.set_xticks(nodes)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, util in zip(bars, utilizations):
            if util > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{util:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Activity breakdown (stacked bar) - use global configuration
        activity_types = ['traf_out', 'traf_in', 'process', 'comp', 'reply_out', 'reply_in', 'local_process']
        colors_list = [get_event_color_and_alpha(activity)[0] for activity in activity_types]
        
        bottom = np.zeros(self.total_nodes)
        
        for i, activity in enumerate(activity_types):
            values = []
            for node_id in range(self.total_nodes):
                activity_time = node_activity_breakdown[node_id].get(activity, 0)
                percentage = (activity_time / self.max_time) * 100 if self.max_time > 0 else 0
                values.append(percentage)
            
            if sum(values) > 0:  # Only plot if there are activities of this type
                ax2.bar(nodes, values, bottom=bottom, label=activity.replace('_', ' ').title(),
                        color=colors_list[i % len(colors_list)], alpha=0.8)
                bottom += values
        
        ax2.set_xlabel('Node ID')
        ax2.set_ylabel('Time (%)')
        ax2.set_title('Node Activity Breakdown')
        ax2.set_xticks(nodes)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def print_event_summary(self):
        """Print a summary of all events"""
        if not self.events:
            print("No events recorded.")
            return
        
        print(f"\n{'='*60}")
        print(f"SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total simulation time: {self.max_time} cycles")
        print(f"Total events: {len(self.events)}")
        print(f"Nodes involved: {self.total_nodes}")
        
        print(f"\n{'Node':<6} {'Events':<8} {'First':<8} {'Last':<8} {'Utilization'}")
        print(f"{'-'*50}")
        
        for node_id in sorted(self.node_states.keys()):
            events = self.node_states[node_id]
            if events:
                first_event = min(event.start_time for event in events)
                last_event = max(event.end_time for event in events)
                total_time = sum(event.end_time - event.start_time for event in events)
                utilization = (total_time / self.max_time) * 100 if self.max_time > 0 else 0
                
                print(f"{node_id:<6} {len(events):<8} {first_event:<8} {last_event:<8} {utilization:.1f}%")
        
        print(f"\n{'Event Type':<15} {'Count':<8} {'Total Time':<12} {'Avg Duration'}")
        print(f"{'-'*50}")
        
        event_stats = defaultdict(list)
        for event in self.events:
            duration = event.end_time - event.start_time
            event_stats[event.event_type].append(duration)
        
        for event_type, durations in event_stats.items():
            count = len(durations)
            total_time = sum(durations)
            avg_duration = total_time / count if count > 0 else 0
            print(f"{event_type:<15} {count:<8} {total_time:<12} {avg_duration:.1f}")

def visualize_simulation(simulator, json_path, timeline_path=None, utilization_path=None):
    """Main function to run simulation with visualization"""
    # Parse config
    arch, workload = simulator.parse_config(json_path)
    
    # Create visualizer and run enhanced simulation
    viz = NoCSimulationVisualizer()
    total_latency = viz.simulate_with_tracking(simulator, workload, arch)
    
    # Print summary
    viz.print_event_summary()
    
    # Create plots
    timeline_fig = viz.plot_timeline(save_path=timeline_path)
    util_fig = viz.plot_node_utilization(save_path=utilization_path)
    
    # Only show plots if no save paths provided (backwards compatibility)
    if not timeline_path and not utilization_path:
        plt.show()
    
    return total_latency, viz

def plot_timeline(json_path, timeline_path=None, utilization_path=None, verbose=True):
    """Convenience function to plot timeline from JSON path"""
    from simulator_stub_analytical_model import FastNoCSimulator
    
    simulator = FastNoCSimulator()
    total_latency, viz = visualize_simulation(
        simulator, 
        json_path, 
        timeline_path=timeline_path, 
        utilization_path=utilization_path
    )
    
    if verbose:
        print(f"Total simulation latency: {total_latency} cycles")
    
    return total_latency, viz

class NoCPlotter:

    def __init__(self):

        self.fig = None # Figure Object
        self.ax = None # Axes Object
        self.topology = None # The topology of the mesh
        self.points = {} # List of the nodes/points
        self.points[0] = []  # NoC elements
        self.points[1] = []  # NPUs
        self.points[2] = []  # Reconfiguring memories
        self.artists_points = {} # List of the artists for the points
        self.artists_points[0] = [] 
        self.artists_points[1] = []
        self.artists_points[2] = []
        self.artists_points["txt"] = [] 
        self.connections = [] # List of the connection between the points 
        self.artists_hconnections = {} # List of the artists for the connections (horizontal)
        self.artists_reconf_connections = {} # List of the artists for the connections between the reconfiguring memories and the PEs
        self.artists_vconnections = {} # List of the artists for the connections (vertical)
        self.num_of_layers = 0
        self.layers = [] # list of the layers
        self.faces = [] # List of the faces, for drawing reasons

    def init(self, config_file):
        """
        Initialize the variables
        """

        def _get_neighbors(k,n,i, topology):
            neighbors = []
            if topology == "mesh":
                if i % k != 0:
                    neighbors.append((i-1, 0))
                if i % k != k-1:
                    neighbors.append((i+1, 0))
                if i // k != 0:
                    neighbors.append((i-k, 0))
                if i // k != k-1:
                    neighbors.append((i+k, 0))
                return neighbors
            elif topology == "torus":
                if i % k != 0: # left neighbor
                    neighbors.append((i-1, 0))
                else:
                    neighbors.append((k-1+i, 1)) # 1 is for left wrap-around
                if i % k != k-1: # right neighbor
                    neighbors.append((i+1, 0))
                else:
                    neighbors.append((i-k+1, 2)) # 2 is for right wrap-around
                if i // k != 0: # bottom neighbor
                    neighbors.append((i-k, 0))
                else:
                    neighbors.append((k*(k-1)+i, 3)) # 3 is for bottom wrap-around
                if i // k != k-1: # top neighbor
                    neighbors.append((i+k, 0))
                else:
                    neighbors.append((i % k, 4)) # 4 is for top wrap-around
                
            else:
                raise RuntimeError('topology type is not supported')
            return neighbors
        
        # read the arch field from configuration file
        config = json.load(open(config_file))
        arch = config['arch']
        self.topology = arch["topology"]
        self.k = arch["k"]
        self.n = arch["n"]
        self.reconf = arch["reconfiguration"]

        # Number of layers
        self.num_of_layers = 2  # one layer for NoC elements, the other for NPUs + one for reconfiguring memories
        if self.reconf != 0:
            self.num_of_layers += 1
        proc_elemnt_ids = []
        for pe in range(arch["k"]** arch["n"]):
            proc_elemnt_ids.append(pe)


        # Points is a list of tuples
        for p in range(arch["k"]** arch["n"]):
            x = p % arch["k"]
            y = p // arch["k"]
            z = 0
            self.points[0].append((x, y, z)) # 0 is for NoC elements
            self.points[1].append((x, y, z+0.5)) # 1 is for NPUs
            # if the field "reconfiguration" is set to any number different from 0,
            # we must also append a third layer representing the reconfiguring memories
            if self.reconf != 0:
                self.points[2].append((x, y, z+1))

        # Build the list of connections based on the mesh topology
        
        for node in range(arch["k"]** arch["n"]):
            neighbors = _get_neighbors(arch["k"], arch["n"], node, self.topology) +[(node, 0)]
            for neighbor in neighbors:
                sorted_connection = sorted([node, neighbor[0]])
                sorted_connection = (tuple(sorted_connection), neighbor[1])
                # check that no other connection with the same vertices is already in the list
                pcheck = False
                for c in self.connections:
                    if c[0] == sorted_connection[0]:
                        pcheck = True
                        break
                if not pcheck:
                    self.connections.append(sorted_connection)
        

        
    ###############################################################################


    def create_fig(self):
        """
        Create the figure object
        """
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1*self.k, 1*self.k, self.k * 0.5])
        self.ax.axis("off")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
    ###############################################################################

    def vertical_connection(self, p1_ix, p2_ix):
        """
        Draws the vertical connection of the points
        """
        x = []
        y = []
        z = []

        x.append(self.points[0][p1_ix][0])
        x.append(self.points[1][p2_ix][0])

        y.append(self.points[0][p1_ix][1])
        y.append(self.points[1][p2_ix][1])

        z.append(self.points[0][p1_ix][2])
        z.append(self.points[1][p2_ix][2])

        artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
        self.artists_vconnections[(p1_ix, p2_ix)] = artist

        if self.reconf != 0:
            x.append(self.points[1][p1_ix][0])
            x.append(self.points[2][p2_ix][0])

            y.append(self.points[1][p1_ix][1])
            y.append(self.points[2][p2_ix][1])

            z.append(self.points[1][p1_ix][2])
            z.append(self.points[2][p2_ix][2])

            artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
            self.artists_reconf_connections[(p2_ix, p2_ix)] = artist


    def horizontal_connection(self, p1_ix, p2_ix, in_plane):
        """
        Draws the horizontal connection of the points
        """
        x = []
        y = []
        z = []
    
        if in_plane == 0:
            x.append(self.points[0][p1_ix][0])
            x.append(self.points[0][p2_ix][0])

            y.append(self.points[0][p1_ix][1])
            y.append(self.points[0][p2_ix][1])

            z.append(self.points[0][p1_ix][2])
            z.append(self.points[0][p2_ix][2])

            # plot only if the connection has not been plotted yet
            artist, = self.ax.plot(x, y, z, color='black', alpha = 0.3)
            self.artists_hconnections[(p1_ix, p2_ix)] = artist

        else:
            # if the in_plane flag is 0 (this can only happen in the torus topology)
            # this means that the considered connection is a wrap-around connection:
            # we represent this by drawing a line from the two points to the border of the mesh

            border = 0.5

            segment1 = [(self.points[0][p1_ix][0], self.points[0][p1_ix][1], self.points[0][p1_ix][2]), (self.points[0][p1_ix][0] + (- border if in_plane == 1 else border if in_plane == 2 else 0), self.points[0][p1_ix][1] + (-border if in_plane == 3 else  border if in_plane == 4 else 0), self.points[0][p1_ix][2])]
            segment2 = [(self.points[0][p2_ix][0], self.points[0][p2_ix][1], self.points[0][p2_ix][2]), (self.points[0][p2_ix][0] + ( border if in_plane == 1 else - border if in_plane == 2 else 0), self.points[0][p2_ix][1] + (border if in_plane == 3 else - border if in_plane == 4 else 0), self.points[0][p2_ix][2])]

            artist1, = self.ax.plot([segment1[0][0], segment1[1][0]], [segment1[0][1], segment1[1][1]], [segment1[0][2], segment1[1][2]], color='black', alpha = 0.4)
            artist2, = self.ax.plot([segment2[0][0], segment2[1][0]], [segment2[0][1], segment2[1][1]], [segment2[0][2], segment2[1][2]], color='black', alpha = 0.4)

            self.artists_hconnections[(p1_ix, p2_ix)] = [artist1, artist2]


    def plot_connections(self):
        """
        Plot the connections between the nodes/points
        """
        for p in range(len(self.points[0])):
            self.vertical_connection(p, p)

        for c in self.connections:
            p1_ix, p2_ix = c[0]
            in_plane= c[1]

            self.horizontal_connection(p1_ix, p2_ix, in_plane)
    ###############################################################################


    def annotate_points(self):
        """
        Annotating the points using their index
        """
        points_coordinates = np.array(self.points[1])
        i = 0
        for x, y, z in zip(points_coordinates[:, 0], points_coordinates[:, 1], points_coordinates[:, 2]):
            self.artists_points["txt"].append(self.ax.text(x, y, z + 0.57 , i, size=8, color='k', fontdict={'weight': 'bold'}, ha='left', va='bottom'))
            i = i + 1

    def plot_nodes(self, points):
        """
        Annotating the points (NoC Nodes)using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "lightseagreen", marker="o", markersize=10, alpha = 0.3)
            self.artists_points[0].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[0].append(self.ax.scatter(xs, ys, zs,color = "lightseagreen", s = 200, alpha = 0.3))#, marker=m)


    def plot_pes(self, points):
        """
        Annotating the points (PEs) using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "tomato", marker="s", markersize=10, alpha = 0.3)
            self.artists_points[1].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[1].append(self.ax.scatter(xs, ys, zs,color = "tomato" , s = 200, marker="s", alpha = 0.3)) 

    def plot_reconf(self, points):
        """
        Annotating the points (Reconfiguring Memories) using their index
        """
        points_coordinates = []
        for p in points:
            points_coordinates.append(p)
            x = p[0]
            y = p[1]
            z = p[2]
            artist, = self.ax.plot(x, y, z, color = "limegreen", marker="D", markersize=10, alpha = 0.3)
            self.artists_points[2].append(artist)
        # points_coordinates = np.array(points_coordinates)
        # xs = points_coordinates[:, 0]
        # ys = points_coordinates[:, 1]
        # zs = points_coordinates[:, 2]
        # self.artists_points[2].append(self.ax.scatter(xs, ys, zs,color = "khaki" , s = 200, marker="D", alpha = 0.3))
    ###############################################################################

    def colorize_nodes(self, currently_active:set, verbose:bool = False):
        """
        Colorize the nodes
        """
        if verbose:
            print("Currently active nodes: ", currently_active)
        for i in range(len(self.points[0])):
            if i in currently_active:
                self.artists_points[0][i].set_alpha(1)
            else:
                self.artists_points[0][i].set_alpha(0.3)

                
    def colorize_pes(self, currently_active_comp:set, currently_active_traf:set, currently_active_reconf: set, verbose: bool = False):
        """
        Colorize the PEs
        """
        if verbose:
            print("Currently active PEs for computation: ", currently_active_comp)
            print("Currently active PEs for traffic: ", currently_active_traf)
        for i in range(len(self.points[1])):
            if i in currently_active_comp:
                assert i not in currently_active_reconf
                self.artists_points[1][i].set_color("tomato")
                self.artists_points[1][i].set_alpha(1)
            elif i in currently_active_traf:
                self.artists_points[1][i].set_color("khaki")
                self.artists_points[1][i].set_alpha(0.8)
            elif i in currently_active_reconf:
                self.artists_points[1][i].set_color("limegreen")
                self.artists_points[1][i].set_alpha(1)
            else:
                self.artists_points[1][i].set_color("tomato")
                self.artists_points[1][i].set_alpha(0.3)

    def colorize_reconf(self, currently_active:set, verbose: bool = False):
        """
        Colorize the reconfiguring memories
        """
        if verbose:
            print("Currently active reconfiguring memories: ", currently_active)
        for i in range(len(self.points[2])):
            if i in currently_active:
                self.artists_points[2][i].set_alpha(1)
            else:
                self.artists_points[2][i].set_alpha(0.3)

    def colorize_connections(self, currently_active:set, verbose: bool = False):
        """
        Colorize the connections
        """
        if verbose:
            print("Currently active connections: ", currently_active)
        for c in self.connections:
            to_check = self.artists_vconnections if c[0][0] == c[0][1] else self.artists_hconnections
            if c[0] in currently_active:
                if isinstance(to_check[c[0]], list):
                    for a in to_check[c[0]]:
                        a.set_alpha(1)
                else:
                    to_check[c[0]].set_alpha(1) 
            else:
                if isinstance(to_check[c[0]], list):
                    for a in to_check[c[0]]:
                        a.set_alpha(0.3)
                        
                else:
                    to_check[c[0]].set_alpha(0.3)

    def colorize_reconf_connections(self, currently_active:set, verbose: bool = False):
        """
        Colorize the connections between the reconfiguring memories and the PEs
        """
        if verbose:
            print("Currently active connections between reconfiguring memories and PEs: ", currently_active)
        #loop over the currently active connections, find the corresponding artists and set their alpha to 1
        for c in [(k,k) for k in range(len(self.points[0]))]:
            if c[0] in currently_active:
                if isinstance(self.artists_reconf_connections[c[0]], list):
                    for a in self.artists_reconf_connections[c[0]]:
                        a.set_alpha(1)
                else:
                    self.artists_reconf_connections[c[0]].set_alpha(1)
            else:
                if isinstance(self.artists_reconf_connections[c[0]], list):
                    for a in self.artists_reconf_connections[c[0]]:
                        a.set_alpha(0.3)
                else:
                    self.artists_reconf_connections[c[0]].set_alpha(0.3)

            
    

    ###############################################################################
    def create_faces(self):
        """
        Create the faces of the mesh, each layer will become a face
        """
        # Make layers
        global layers
        global topology

        # Separate lists of x, y and z coordinates
        x_s = []
        y_s = []
        z_s = []

        for i in range(0, self.num_of_layers):
            layer = []

            for p in self.points[i]:
                    layer.append(p)
                    x_s.append(p[0])
                    y_s.append(p[1])
                    if (p[2] not in z_s):
                        z_s.append(p[2])

            self.layers.append(layer)

        # Making faces, only out of the corner points of the layer
        if self.topology == "mesh":
            for i in range(0, self.num_of_layers):
                x_min = min(x_s)
                x_max = max(x_s)
                y_min = min(y_s)
                y_max = max(y_s)
                face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
                self.faces.append(face)
        elif self.topology == "torus":
            added_border = 0.5
            for i in range(0, self.num_of_layers):
                x_min = min(x_s) - added_border
                x_max = max(x_s) + added_border
                y_min = min(y_s) - added_border
                y_max = max(y_s) + added_border
                face = list([(x_min, y_min, z_s[i]), (x_max, y_min, z_s[i]), (x_max, y_max, z_s[i]), (x_min, y_max, z_s[i])])
                self.faces.append(face)
    ###############################################################################


    def plot_faces(self):
        """
        Plot the faces
        """
        # Create multiple 3D polygons out of the faces
        poly = Poly3DCollection(self.faces, linewidths=1, alpha=0.1)
        faces_colors = []
        # Generating random color for each face
        for i in range(0, self.num_of_layers):
            red = np.random.randint(0, 256)
            green = np.random.randint(0, 256)
            blue = np.random.randint(0, 256)
            color = (red, green, blue)
            if color not in faces_colors:
                faces_colors.append('#%02x%02x%02x' % color)
        poly.set_facecolors(faces_colors)
        self.ax.add_collection3d(poly)
    ###############################################################################

    def gen_activity_animation(self, logger, fps: int = 100, file_name: Union[str, None] = None, verbose: bool = True):
        """
        The function takes as input a logger object (defined in the nocsim module, exposed in restart). This is used as a chonological timeline
        on which the events that happen on NoC are registered. We unroll this timeline and plot it in a matplotlib animation

        Args:
        - logger: nocsim.EventLogger object
        - fine_name: a string, 

        Returns:
        - None
        """

        anti_events_map = { nocsim.EventType.IN_TRAFFIC : nocsim.EventType.OUT_TRAFFIC,
                            nocsim.EventType.END_COMPUTATION : nocsim.EventType.START_COMPUTATION,
                            nocsim.EventType.END_RECONFIGURATION : nocsim.EventType.START_RECONFIGURATION,
                            nocsim.EventType.END_SIMULATION : nocsim.EventType.START_SIMULATION}

        self.timeStamp = self.ax.text(0, 0, 1., 0, size=12, color='red')
        cycles = logger.events[-1].cycle
        events_pointer = 0 # pointer to the events in the logger
        current_events = set() # set of the current events

        def _init_graph():
            nonlocal events_pointer, current_events
            events_pointer = 0
            current_events = set()

        def _update_graph(cycle):
            """
            Update the graph at each cycle
            """
            nonlocal events_pointer, current_events, anti_events_map, verbose
            if verbose:
                print(f"--- Cycle: {cycle} ---")
                print(f"Events pointer: {events_pointer}")
            #for each cycle, compare it with the starting cycle of the event
            while cycle >= logger.events[events_pointer].cycle:
                if (logger.events[events_pointer].type in anti_events_map.values()):
                        current_events.add(events_pointer)
                        events_pointer += 1
                else:
                    # find the event in the current events and remove it
                    event_type = anti_events_map[logger.events[events_pointer].type]
                    additional_info = logger.events[events_pointer].additional_info
                    ctype = logger.events[events_pointer].ctype
                    event_to_remove = []
                    for event in current_events:
                        if (logger.events[event].type == event_type and
                            logger.events[event].additional_info == additional_info and
                            logger.events[event].ctype == ctype):
                            
                            assert logger.events[event].cycle <= logger.events[events_pointer].cycle
                            # remove the event from the current events
                            event_to_remove.append(event)
                            break
                    
                    if len(event_to_remove) == 0:
                        raise RuntimeError(f"Event {events_pointer} not found in the current events")
                    for event in event_to_remove:
                        current_events.remove(event)
                    events_pointer += 1       


            # loop over the current events and update the graph
            currently_active_nodes = set()
            currently_active_pes_comp = set()
            currently_active_pes_reconf = set()
            currently_active_pes_traf = set()
            currently_active_connections = set()
            for event in current_events:
                if logger.events[event].type == nocsim.EventType.OUT_TRAFFIC:
                    # check what type of channel is active at the moment
                    hystory = logger.events[event].info.history
                    for h in hystory:
                        # find the h element such that h[3]>= cycle and h4 <= cycle
                        if h.start <= cycle and h.end > cycle:
                            currently_active_connections.add(tuple(sorted([h.rsource, h.rsink])))
                            currently_active_nodes.add(h.rsource)
                            currently_active_nodes.add(h.rsink)
                            if h.rsource == h.rsink:
                                currently_active_pes_traf.add(h.rsource)
                            break
                        elif h.start > cycle:
                            # the connection is not active, the packet is still being processed by the source
                            currently_active_nodes.add(h.rsource)
                            break   

                elif logger.events[event].type == nocsim.EventType.START_COMPUTATION:
                    currently_active_pes_comp.add(logger.events[event].info.node)
                elif logger.events[event].type == nocsim.EventType.START_RECONFIGURATION:
                    currently_active_pes_reconf.add(logger.events[event].additional_info)
                    

            self.colorize_nodes(currently_active_nodes, verbose)
            self.colorize_pes(currently_active_pes_comp, currently_active_pes_traf, currently_active_pes_reconf, verbose)
            self.colorize_connections(currently_active_connections, verbose)
            self.colorize_reconf(currently_active_pes_reconf, verbose)
            self.timeStamp.set_text(f"Cycle: {cycle}")

            # plt.draw()
        # Crea l'animazione utilizzando FuncAnimation
        ani = FuncAnimation(self.fig, _update_graph, frames=range(cycles), init_func=_init_graph ,repeat=False, interval=1000/fps)

        # Salva l'animazione se file_name è specificato
        if file_name:
            ani.save(file_name, writer='imagemagick', fps=fps, dpi= 100)

        plt.show()
            
            
    ###############################################################################

    def plot(self,logger, pause, network_file = None, file_name = None, verbose = False):
        """
        Main Execution Point
        """
        curdir = os.path.dirname(__file__)
        network_file = curdir +'/../config_files/arch.json' if network_file is None else network_file

        try:
            network_file = sys.argv[1]
        except IndexError:
            pass
        self.init(network_file)
        self.create_fig()
        self.plot_connections()
        self.annotate_points()
        # create_faces()
        # plot_faces()
        self.plot_nodes(self.points[0])
        self.plot_pes(self.points[1])
        self.plot_reconf(self.points[2])
        self.gen_activity_animation(logger, pause,file_name, verbose)
        #plt.show()
    ###############################################################################

class NoCTimelinePlotter(NoCPlotter):
    """Subclass for 2D timeline visualization. Inherits all NoCPlotter methods."""
    
    def __init__(self):
        super().__init__()
        self.fig2d = None  # Separate figure for 2D timeline
        self.ax2d = None
        self.node_events = {}  # Stores event data: {node: {"comp": [(start, duration)], "traf": [...]}}

    def setup_timeline(self, logger, config_file):
        """Initialize 2D timeline figure and preprocess events."""
        # Initialize parent class with architecture config
        self.init(config_file)  # Loads node data from config_file
        self._preprocess_events(logger)
        self.fig2d, self.ax2d = plt.subplots(figsize=(10, 6))
        self.ax2d.set_xlabel("Cycle")
        self.ax2d.set_ylabel("Node")
        self.ax2d.grid(False)
        self.max_cycle = logger.events[-1].cycle

    def _preprocess_events(self, logger):
        """ Extract computation/traffic events per node.
            Information about Computing, Traffic (out and in) and Reconfiguration 
            is collected from the logger
        """
        self.node_events = {i: {"comp": [], "recon": [], "traf_out": [], "traf_in": [], "traf_between":[], "reply_in": [], "reply_out": []} for i in range(len(self.points[1]))}
    
        # Process computation events
        for event in logger.events:
            ######### COMPUTATIONS #######
            if event.type == nocsim.EventType.START_COMPUTATION:
                node = event.info.node
                start = event.cycle
                
                # Find matching END_COMPUTATION with error handling
                try:
                    end_event = next(
                        e for e in logger.events 
                        if e.type == nocsim.EventType.END_COMPUTATION 
                        and e.info.node == node
                        and e.cycle > start  # Ensure valid duration
                    )
                    duration = end_event.cycle - start
                    self.node_events[node]["comp"].append((start, duration))
                except StopIteration:
                    print(f"Warning: No END_COMPUTATION found for node {node} at cycle {start}")
                    continue 
            ######### TRAFFIC #######    
            elif event.type == nocsim.EventType.OUT_TRAFFIC:
                #the same information is stored in OUT_TRAFFIC and IN_TRAFFIC
                #note from Edoardo: 
                # because it is used to record when the packet actually arrives in the chronologiacal log: 
                # both events (OUT_TRAFFIC and IN_TRAFFIC) point to the same info, which records the hystory of the related traffic, 
                # whose TrafficEventInfo is also pointed by both
                #Here we want to collect: sending node, receiving node, time of sending, time of receiving
                #Communication between the nodes is denoted with different color
                
                id_message = event.additional_info
                communication_type = event.ctype #comunication type of the event
                start = event.cycle
                sending_node = event.info.source
                receiving_node = event.info.dest
                
                duration = 0
                for history_bit in event.info.history:
                    if history_bit.start >= start and history_bit.end >= start:
                        
                        if history_bit.rsource == history_bit.rsink == sending_node:
                            duration = history_bit.end - history_bit.start 
                            self.node_events[history_bit.rsource]["traf_out" if communication_type != 4 else "reply_out"].append((history_bit.start, duration))
                            
                        elif history_bit.rsource == history_bit.rsink == receiving_node:
                            duration = history_bit.end - history_bit.start 
                            self.node_events[history_bit.rsink]["traf_in" if communication_type != 4 else "reply_in"].append((history_bit.start, duration))
                        
                        elif history_bit.rsource != history_bit.rsink:
                            duration = history_bit.end - history_bit.start
                            # self.node_events[history_bit.rsource]["traf_between"].append((history_bit.start, duration))
                            # self.node_events[history_bit.rsink]["traf_between"].append((history_bit.start, duration))
                        else: 
                            raise ValueError(f"Error: I don't know what to do with this history bit {history_bit} of {event} ")
                    else:
                        raise ValueError(f"Error: No history bit found for OUT_TRAFFIC event at cycle {start}")
            
            ######### RECONFIGURATION #######
            elif event.type == nocsim.EventType.START_RECONFIGURATION:
                node = event.additional_info
                start = event.cycle
                
                # Find matching END_RECONFIGURATION with error handling
                try:
                    end_event = next(
                        e for e in logger.events 
                        if e.type == nocsim.EventType.END_RECONFIGURATION 
                        and e.additional_info== node
                        and e.cycle > start  # Ensure valid duration
                    )
                    duration = end_event.cycle - start
                    self.node_events[node]["recon"].append((start, duration))
                except StopIteration:
                    print(f"Warning: No END_RECONFIGURATION found for node {node} at cycle {start}")
                    continue 
            ######### Events to omit #####
            elif event.type == nocsim.EventType.START_SIMULATION or nocsim.EventType.END_SIMULATION:
                continue
            else:
                raise TypeError(f"Unknown event type: {event.type}") 
                        
    
    def _print_node_events(self):
        """Print event data for debugging."""
        #node_events[node]["comp"]
        for node, events in self.node_events.items():
            print(f"Node {node}:")
            print(f"Computation events and duration: {events['comp']}")
            print(f"Traffic events IN and duration: {events['traf_in']}")
            print(f"Traffic events OUT and duration: {events['traf_out']}")
            print(f"Traffic events BETWEEN and duration: {events['traf_between']}")
            print(f"Reconfiguration events and duration: {events['recon']}")
            print()

    def plot_timeline(self, filename, legend=True, hihlight_xticks=True):
        """Draw horizontal bars for events."""
        
        # Create event_types from global configuration
        event_type_labels = {
            'comp': 'Computation',
            'recon': 'Reconfiguration', 
            'traf_out': 'Traffic PE out',
            'traf_in': 'Traffic PE in',
            'traf_between': 'Traffic NoC',
            'reply_in': 'Reply PE in',
            'reply_out': 'Reply PE out'
        }
        event_types = [
            (event_type, color, event_type_labels[event_type], alpha)
            for event_type, (color, alpha) in GLOBAL_EVENT_COLORS.items()
            if event_type in event_type_labels
        ]
    
    
        # Track used labels to prevent duplicates in legend
        used_labels = set()

        for node, events in self.node_events.items():
            has_events = False  # Flag to check if node has any events
            
            for event_key, color, label, alpha in event_types:
                event_list = events.get(event_key, [])
                if event_list:
                    has_events = True
                    # Use label only if it hasn't been used before
                    current_label = label if label not in used_labels else None
                    self.ax2d.broken_barh(
                        event_list,
                        (node - 0.4, 0.8),
                        facecolors=color,
                        label=current_label,
                        alpha=alpha
                    )
                    if current_label:
                        used_labels.add(label)
            
            # Handle nodes with no events
            if not has_events:
                print(f"No events found for node {node}")
        
        # Set y-ticks to node IDs
        self.ax2d.set_yticks(range(len(self.points[1])))
        #for debug purposes, only show first 2 nodes
        #self.ax2d.set_yticks(range(2))
        
        # set x-ticks to cycle numbers
        self.ax2d.set_xlim(0, self.max_cycle)
        #self.ax2d.set_xlim(0, 120)
        # Auto-adjust ticks
        self.ax2d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        self.ax2d.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        # Add vertical lines at each major x-tick
        if hihlight_xticks:
            for tick in self.ax2d.xaxis.get_major_locator().tick_values(self.ax2d.get_xlim()[0], self.ax2d.get_xlim()[1]):
                self.ax2d.axvline(x=tick, color='grey', linestyle='-', linewidth=0.5, zorder = 0)

        # Add horizontal lines at the corners of the x nodes
        for node in range(len(self.points[1])):
            self.ax2d.axhline(y=node - 0.4, color='grey', linestyle='--', linewidth=0.5)
            self.ax2d.axhline(y=node + 0.4, color='grey', linestyle='--', linewidth=0.5)

        if legend:
            self.ax2d.legend(
            loc='upper center',
            ncol=6,
            bbox_to_anchor=(0.5, 1.05),
            #bbox_to_anchor=(1.17, 1.0),
            #borderaxespad=0.0,
            #fontsize='small',
            #title='Event Types',
            fancybox=True,
            shadow=True,
            title_fontsize='medium',
            frameon=True
            )
        if filename: 
            self.fig2d.savefig(filename, dpi=300)
            print(f"Timeline graph saved to {filename}")
        plt.close(self.fig)
        
        
        
    def plot_timeline_with_debug_annotations(self, filename, legend=True, hihlight_xticks=True, annoate_events = True):
        """Draw horizontal bars for events."""
        
        # Create event_types from global configuration
        event_type_labels = {
            'comp': 'Computation',
            'recon': 'Reconfiguration', 
            'traf_out': 'Traffic PE out',
            'traf_in': 'Traffic PE in',
            'traf_between': 'Traffic NoC',
            'reply_in': 'Reply PE in',
            'reply_out': 'Reply PE out'
        }
        event_types = [
            (event_type, color, event_type_labels[event_type], alpha)
            for event_type, (color, alpha) in GLOBAL_EVENT_COLORS.items()
            if event_type in event_type_labels
        ]

        used_labels = set()

        for node, events in self.node_events.items():
            has_events = False

            for event_key, color, label, alpha in event_types:
                event_list = events.get(event_key, [])
                if event_list:
                    has_events = True
                    
                    # Extract (start, duration) for plotting
                    plot_event_list = [(e["start"], e["duration"]) for e in event_list]
                    
                    # Add bars
                    current_label = label if label not in used_labels else None
                    if event_key == "comp":
                        self.ax2d.broken_barh(
                            plot_event_list,
                            (node - 0.4, 0.8),
                            facecolors=color,
                            label=current_label,
                            alpha=alpha,
                            edgecolor='black', #add line around rectangle
                            linewidth=0.5,
                            linestyle='-'
                            )
                    else: 
                        self.ax2d.broken_barh(
                            plot_event_list,
                            (node - 0.4, 0.8),
                            facecolors=color,
                            label=current_label,
                            alpha=alpha,
                            edgecolor='black', #add line around rectangle
                            linewidth=0.5,
                            linestyle='--'
                            )
                        
                    if current_label:
                        used_labels.add(label)

                    if annoate_events:
                        # Add text annotations
                        for event in event_list:
                            mid_point = event["start"] + event["duration"] / 2
                            
                            # Customize text based on event type
                            if event_key == "comp":
                                text = f"Comp\nID:{event['id_task']}"  # Comp shows only task ID
                            elif event_key in ["traf_out", "traf_in", "traf_between", "reply_in", "reply_out"]:
                                text = f"Mess\nID:{event['id_task']}\n({event['com_type']})"  # Traffic: ID + type
                            else:
                                text = None  # Skip recon/others unless needed
                            
                            if text:
                                self.ax2d.text(
                                    mid_point,
                                    node,
                                    text,
                                    ha='center',
                                    va='center',
                                    color='black',
                                    fontsize=8,
                                    clip_on=True
                                )

            if not has_events:
                print(f"No events found for node {node}")
        
        # Set y-ticks to node IDs
        self.ax2d.set_yticks(range(len(self.points[1])))
        #for debug purposes, only show first 2 nodes
        #self.ax2d.set_yticks(range(2))
        
        # set x-ticks to cycle numbers
        self.ax2d.set_xlim(0, self.max_cycle)
        # Auto-adjust ticks
        self.ax2d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        self.ax2d.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        # Add vertical lines at each major x-tick
        if hihlight_xticks:
            for tick in self.ax2d.xaxis.get_major_locator().tick_values(self.ax2d.get_xlim()[0], self.ax2d.get_xlim()[1]):
                self.ax2d.axvline(x=tick, color='grey', linestyle='-', linewidth=0.5, zorder = 0)

        # Add horizontal lines at the corners of the x nodes
        for node in range(len(self.points[1])):
            self.ax2d.axhline(y=node - 0.4, color='grey', linestyle='--', linewidth=0.5)
            self.ax2d.axhline(y=node + 0.4, color='grey', linestyle='--', linewidth=0.5)

        if legend:
            self.ax2d.legend(
            #loc='upper right',
            #bbox_to_anchor=(1.17, 1.0),
            #borderaxespad=0.0,
            #fontsize='small',
            #title='Event Types',
            title_fontsize='medium',
            frameon=True
            )
        if filename: 
            self.fig2d.savefig(filename, dpi=300)
            print(f"Timeline graph saved to {filename}")
        plt.close(self.fig)
        
    '''      
    def plot_timeline_factor_back(self, filename, factor_comp, factor_recon, legend=True, hihlight_xticks=True):
        """Draw horizontal bars for events."""
        
        # Create event_types from global configuration
        event_type_labels = {
            'comp': 'Computation',
            'recon': 'Reconfiguration', 
            'traf_out': 'Traffic PE out',
            'traf_in': 'Traffic PE in',
            'traf_between': 'Traffic NoC',
            'reply_in': 'Reply PE in',
            'reply_out': 'Reply PE out'
        }
        event_types = [
            (event_type, color, event_type_labels[event_type], alpha)
            for event_type, (color, alpha) in GLOBAL_EVENT_COLORS.items()
            if event_type in event_type_labels
        ]
        
        # Create factor mapping and initialize data structures
        event_factors = {
            'comp': factor_comp,
            'recon': factor_recon,
            'traf_out': 1.0,
            'traf_in': 1.0,
            'traf_between': 1.0,
            'reply_in': 1.0,
            'reply_out': 1.0
        }
        
        # reverted_events = {}
        # max_cycle = 0

        # # Process each node independently
        # for node, node_events in self.node_events.items():
        #     # Sort events by original sped-up start time
        #     sorted_events = []
        #     for event_key, events in node_events.items():
        #         for start, duration in events:
        #             sorted_events.append((start, duration, event_key))
        #     sorted_events.sort(key=lambda x: x[0])
            
        #     reverted_node_events = {}
        #     cumulative_shift = 0.0
        #     last_end = 0.0
            
        #     for start_spedup, duration_spedup, event_key in sorted_events:
        #         factor = event_factors[event_key]
                
        #         # Calculate real start time accounting for previous scaling
        #         real_start = start_spedup + cumulative_shift
                
        #         if event_key in ['comp', 'recon']:
        #             # Scale the event
        #             real_duration = duration_spedup * factor
        #             # Calculate how much this event stretches the timeline
        #             stretch = (real_duration - duration_spedup)
        #             cumulative_shift += stretch
        #         else:
        #             # Non-scaled event keeps original duration
        #             real_duration = duration_spedup
                
        #         # Store the reverted event
        #         if event_key not in reverted_node_events:
        #             reverted_node_events[event_key] = []
        #         reverted_node_events[event_key].append((real_start, real_duration))
                
        #         # Update maximum cycle
        #         max_cycle = max(max_cycle, real_start + real_duration)
        #         last_end = real_start + real_duration
            
        #     reverted_events[node] = reverted_node_events
        
        # Global timeline processing
        global_timeline = []
        
        # 1. Collect all events from all nodes with node info
        for node, node_events in self.node_events.items():
            for event_key, events in node_events.items():
                for start, duration in events:
                    global_timeline.append((start, duration, event_key, node))
        
        # 2. Sort events by original sped-up start time globally
        global_timeline.sort(key=lambda x: x[0])
        
        # 3. Process events in global order with cumulative shifts
        reverted_events = {}
        max_cycle = 0
        cumulative_shift = 0.0
        last_end = 0.0

        for start_spedup, duration_spedup, event_key, node in global_timeline:
            factor = event_factors[event_key]
            
            # Calculate real start time accounting for global shifts
            real_start = start_spedup + cumulative_shift
            
            if event_key in ['comp', 'recon']:
                # Scale the event
                real_duration = duration_spedup * factor
                # Track timeline stretching
                stretch = real_duration - duration_spedup
                cumulative_shift += stretch
            else:
                # Preserve duration for non-scaled events
                real_duration = duration_spedup
            
            # Update node's event list
            if node not in reverted_events:
                reverted_events[node] = {}
            reverted_events[node].setdefault(event_key, []).append((real_start, real_duration))
            
            # Track maximum cycle and last event end
            max_cycle = max(max_cycle, real_start + real_duration)
            last_end = max(last_end, real_start + real_duration)

        used_labels = set()
        for node in self.node_events:
            has_events = False
            for event_key, color, label, alpha in event_types:
                events = reverted_events.get(node, {}).get(event_key, [])
                
                if events:
                    has_events = True
                    current_label = label if label not in used_labels else None
                    
                    self.ax2d.broken_barh(
                        events,
                        (node - 0.4, 0.8),
                        facecolors=color,
                        label=current_label,
                        alpha=alpha
                    )
                    
                    if current_label:
                        used_labels.add(label)
            
            if not has_events:
                print(f"No events found for node {node}")
        
        self.ax2d.set_yticks(range(len(self.points[1])))
        self.ax2d.set_xlim(0, max_cycle)
        self.ax2d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        self.ax2d.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        
        if hihlight_xticks:
            for tick in self.ax2d.xaxis.get_major_locator().tick_values(self.ax2d.get_xlim()[0], self.ax2d.get_xlim()[1]):
                self.ax2d.axvline(x=tick, color='grey', linestyle='-', linewidth=0.5, zorder=0)

        for node in range(len(self.points[1])):
            self.ax2d.axhline(y=node - 0.4, color='grey', linestyle='--', linewidth=0.5)
            self.ax2d.axhline(y=node + 0.4, color='grey', linestyle='--', linewidth=0.5)

        if legend:
            self.ax2d.legend(
                loc='upper center',
                ncol=6,
                bbox_to_anchor=(0.5, 1.05),
                fancybox=True,
                shadow=True,
                title_fontsize='medium',
                frameon=True
            )
            
        if filename: 
            self.fig2d.savefig(filename, dpi=300)
            print(f"Timeline graph saved to {filename}")
        plt.close(self.fig)   
        '''
class SynchronizedNoCAnimator:
    """Handles synchronized 3D NoC + 2D timeline animation."""
    
    def __init__(self, noc_plotter, timeline_plotter, logger, config_file):
        self.noc_plotter = noc_plotter
        self.timeline_plotter = timeline_plotter
        self.logger = logger
        self.config_file = config_file
        self.current_cycle = 0
        self.max_cycle = logger.events[-1].cycle
        
        # Animation state
        self.events_pointer = 0
        self.current_events = set()
        
        # Create combined figure
        self.fig = plt.figure(figsize=(20, 8))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax2d = self.fig.add_subplot(122)
        
        # Initialize visualizations
        self._add_timeline_elements()
        self._initialize_plotters()
        

    def _initialize_plotters(self):
        """Initialize both visualizations and collect blit artists."""
        # Initialize 3D plot
        self.noc_plotter.init(self.config_file)
        self.noc_plotter.fig = self.fig
        self.noc_plotter.ax = self.ax3d
        self.noc_plotter.plot_connections()
        self.noc_plotter.annotate_points()
        self.noc_plotter.plot_nodes(self.noc_plotter.points[0])
        self.noc_plotter.plot_pes(self.noc_plotter.points[1])
        self.noc_plotter.plot_reconf(self.noc_plotter.points[2])
        
        self.noc_plotter.timeStamp = self.ax3d.text(0, 0, 1., 0, size=12, color='red', 
                                              fontdict={'weight': 'bold'}, 
                                              transform=self.ax3d.transAxes)

        # Initialize timeline
        self.timeline_plotter.setup_timeline(self.logger, self.config_file)
        self.timeline_plotter.ax2d = self.ax2d 
        #Timeliene without legedn, no higlight of the x thick, annotaiton of events - YES
        self.timeline_plotter.plot_timeline(None,legend = False, hihlight_xticks = False, annoate_events = True)
        
        # Collect artists that need blitting
        self.blit_artists = [self.current_line]
        
        # Handle connection artists
        for conn_group in [self.noc_plotter.artists_hconnections,
                         self.noc_plotter.artists_vconnections,
                         self.noc_plotter.artists_reconf_connections]:
            for conn in conn_group.values():
                if isinstance(conn, list):
                    self.blit_artists.extend(conn)
                else:
                    self.blit_artists.append(conn)
                
        self.blit_artists += [
            *self.noc_plotter.artists_points[0],
            *self.noc_plotter.artists_points[1],
            *self.noc_plotter.artists_points[2],
            self.noc_plotter.timeStamp
        ]
        
        # Collect all text artists from the timeline
        timeline_text_artists = self.ax2d.texts  # Get all text objects in the timeline
        self.blit_artists.extend(timeline_text_artists)  # Add them to blit list
        
    def _init_anim(self):
        """Initialization function for blitting"""
        return self.blit_artists


    def _add_timeline_elements(self):
        """Add timeline animation elements"""
        self.current_line = self.ax2d.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        self.ax2d.set_xlim(0, int(min(20, self.max_cycle))) #here integers?
        #self.ax2d.set_xticks(np.arange(0, self.max_cycle + 1, 1))
        self.fig.tight_layout()

    def _process_events(self, cycle):
        """Replicated event processing logic from original _update_graph"""
        # Process events for current cycle
        while self.events_pointer < len(self.logger.events) and cycle >= self.logger.events[self.events_pointer].cycle:
            event = self.logger.events[self.events_pointer]
            anti_events_map = {
                nocsim.EventType.IN_TRAFFIC: nocsim.EventType.OUT_TRAFFIC,
                nocsim.EventType.END_COMPUTATION: nocsim.EventType.START_COMPUTATION,
                nocsim.EventType.END_RECONFIGURATION: nocsim.EventType.START_RECONFIGURATION,
                nocsim.EventType.END_SIMULATION: nocsim.EventType.START_SIMULATION
            }
            
            if event.type in anti_events_map.values():
                self.current_events.add(self.events_pointer)
            else:
                event_type = anti_events_map[event.type]
                additional_info = event.additional_info
                ctype = event.ctype
                to_remove = next(
                    (e for e in self.current_events 
                     if self.logger.events[e].type == event_type 
                     and self.logger.events[e].additional_info == additional_info 
                     and self.logger.events[e].ctype == ctype),
                    None
                )
                if to_remove is not None:
                    self.current_events.remove(to_remove)
            self.events_pointer += 1

    def _update_combined(self, frame):
        """Combined update function for both visualizations"""
        self.current_cycle = frame
        
        # Process events for current cycle
        self._process_events(frame)
        
        # Update 3D visualization using original colorize methods
        currently_active_nodes = set()
        currently_active_pes_comp = set()
        currently_active_pes_reconf = set()
        currently_active_pes_traf = set()
        currently_active_connections = set()

        for event_idx in self.current_events:
            event = self.logger.events[event_idx]
            if event.type == nocsim.EventType.OUT_TRAFFIC:
                for h in event.info.history:
                    if h.start <= frame and h.end > frame:
                        currently_active_nodes.add(h.rsource)
                        currently_active_nodes.add(h.rsink)
                        currently_active_connections.add(tuple(sorted([h.rsource, h.rsink])))
                        if h.rsource == h.rsink:
                            currently_active_pes_traf.add(h.rsource)
                        break
                    elif h.start > frame:
                        currently_active_nodes.add(h.rsource)
                        break
            elif event.type == nocsim.EventType.START_COMPUTATION:
                currently_active_pes_comp.add(event.info.node)
            elif event.type == nocsim.EventType.START_RECONFIGURATION:
                currently_active_pes_reconf.add(event.additional_info)

        self.noc_plotter.colorize_nodes(currently_active_nodes)
        self.noc_plotter.colorize_pes(currently_active_pes_comp, 
                                    currently_active_pes_traf, 
                                    currently_active_pes_reconf)
        self.noc_plotter.colorize_connections(currently_active_connections)
        self.noc_plotter.colorize_reconf(currently_active_pes_reconf)
        self.noc_plotter.timeStamp.set_text(f"Cycle: {frame}")
            
        
        # Update timeline visualization
        self.current_line.set_xdata([frame, frame])
        # ===================================================================
        # x-axis limits and ticks
        # ===================================================================
        window_size = 20  # Total cycles to show in the timeline
        
        # Calculate left/right bounds dynamically
        left = max(0, frame - window_size//2)  # Center the frame in the window
        right = left + window_size
        
        # Handle edge cases where window exceeds simulation duration
        if right > self.max_cycle:
            right = self.max_cycle
            left = max(0, right - window_size)  # Show last 'window_size' cycles
        
        # Apply the calculated bounds
        self.ax2d.set_xlim(left, right)
        self.ax2d.set_xticks(np.arange(left, right + 1, 1))  # Ticks every cycle
        # ===================================================================

            
        #self.ax2d.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        return self.blit_artists

    def create_animation(self, filename, fps=30):
        """Create and save synchronized animation"""
        # Configure FFmpeg writer properly
        writer = FFMpegWriter(
            fps=fps,
            extra_args=['-preset', 'veryslow', '-crf', '18']
        )
        
        ani = FuncAnimation(
            self.fig,
            self._update_combined,
            frames=range(self.max_cycle),
            init_func=self._init_anim,
            interval=1000//fps,
            blit=True
        )

        if filename:
            ani.save(
                filename,
                writer=writer, 
                dpi=150,
                savefig_kwargs={'facecolor': 'white'}
            )
            print(f"Combined animation saved to {filename}")
        
        plt.close(self.fig)




