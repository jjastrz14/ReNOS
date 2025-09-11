import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import seaborn as sns
import networkx as nx

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
            event_type='compute_start',
            description=f"Start COMP_OP task {task_id} (cycles: {ct_required})"
        ))
    
    def track_comp_op_end(self, task_id: int, node: int, end_time: int) -> None:
        """Track the completion of a COMP_OP operation"""
        #print(f"DEBUG: track_comp_op_end - task_id={task_id}, node={node}, end_time={end_time}")
        # Find and update the existing start event
        updated = False
        for event in self.events:
            if event.task_id == task_id and event.node_id == node and event.event_type == 'compute_start':
                event.end_time = end_time
                event.event_type = 'compute'
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
            event_type='send_start',
            related_node=dst,
            size=size,
            description=f"Start sending WRITE to node {dst} (size: {size})"
        ))
        
    def track_write_send_end(self, task_id: int, src: int, dst: int, end_time: int) -> None:
        """Track the end of sending a WRITE message"""
        # Find and update the existing send start event
        for event in self.events:
            if event.task_id == task_id and event.node_id == src and event.event_type == 'send_start':
                event.end_time = end_time
                event.event_type = 'send'
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
            event_type='receive',
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
            event_type='send_reply',
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
            event_type='receive_reply',
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
            if event.task_id == task_id and event.node_id == src and event.event_type == 'send_reply':
                event.end_time = end_time
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
                event_type='compute',
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
            event_type='send',
            related_node=dst,
            size=total_size,
            description=f"Batched WRITE to node {dst} (batch of {len(task_ids)} writes, total size: {total_size})"
        ))

    
    def add_event(self, event: NodeEvent):
        """Add an event to tracking"""
        self.events.append(event)
        self.node_states[event.node_id].append(event)
    
    def plot_timeline(self, figsize=(15, 10), save_path=None):
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
        
        # Color mapping for different event types
        colors = {
            'send': '#0066CC',          # Blue - All traffic from the node
            'receive': '#4ECDC4',       # Keep existing
            'process': '#45B7D1',       # Keep existing  
            'compute': '#FF6347',       # Tomato - Compute operations
            'send_reply': '#FFD700',    # Yellow - All reply going out of the node
            'receive_reply': '#40E0D0', # Turquoise - All reply IN
            'local_process': '#98D8C8'  # Keep existing
        }
        
        y_positions = {}
        current_y = 0
        
        # Sort events by node for cleaner visualization
        for node_id in sorted(self.node_states.keys()):
            y_positions[node_id] = current_y
            current_y += 1
        
        # Plot events
        for event in self.events:
            y_pos = y_positions[event.node_id]
            width = event.end_time - event.start_time
            
            rect = Rectangle(
                (event.start_time, y_pos - 0.4),
                width,
                0.8,
                facecolor=colors.get(event.event_type, '#95A5A6'),
                edgecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # you can add text labels if the rectangle is wide enough
        
        # Customize plot
        ax.set_xlim(0, self.max_time * 1.05)
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        ax.set_xlabel('Time (cycles)', fontsize=12)
        ax.set_ylabel('Node ID', fontsize=12)
        ax.set_title('NoC Simulation Timeline - Node Activities', fontsize=14, fontweight='bold')
        
        # Set y-axis ticks to node IDs
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([f'Node {node_id}' for node_id in sorted(y_positions.keys())])
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Create legend with clearer labels
        legend_labels = {
            'send': 'Send Traffic',
            'receive': 'Receive Traffic',
            'process': 'Process Data',
            'compute': 'Compute Operation',
            'send_reply': 'Send Reply',
            'receive_reply': 'Receive Reply',
            'local_process': 'Local Process'
        }
        
        legend_elements = [
            patches.Patch(color=color, label=legend_labels.get(event_type, event_type.replace('_', ' ').title()))
            for event_type, color in colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
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
        
        # Plot 2: Activity breakdown (stacked bar)
        activity_types = ['send', 'receive', 'process', 'compute', 'send_reply', 'receive_reply', 'local_process']
        colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
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

# Example usage function
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

# Example of how to use it:
#if __name__ == "__main__":
#    from simulator_stub_analytical_model import FastNoCSimulator  # Import your simulator
#    
#    simulator = FastNoCSimulator()
#    total_latency, visualizer = visualize_simulation(simulator, "./data/ACO_single_conv_anal_2_2025-09-04_11-39-59/best_solution.json")
#    
#    print(f"Total simulation latency: {total_latency} cycles")