import json
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
import time

@dataclass
class ArchConfig:
    """Architecture configuration parsed from JSON"""
    topology: str
    width_phit: int
    k: int  # network dimension
    n: int  # nodes per dimension
    routing_function: str
    num_vcs: int
    vc_buf_size: int
    flit_size: int
    routing_delay: int
    vc_alloc_delay: int
    sw_alloc_delay: int
    st_final_delay: int
    reconf_cycles: float
    ANY_comp_cycles: float
    logger: int

    
    @classmethod
    def from_dict(cls, arch_dict: Dict) -> 'ArchConfig':
        return cls(
            topology=arch_dict.get('topology', 'torus'),
            width_phit=arch_dict.get('width_phit', 64),
            k=arch_dict.get('k', 2),
            n=arch_dict.get('n', 2),
            routing_function=arch_dict.get('routing_function', 'dim_order'),
            num_vcs=arch_dict.get('num_vcs', 16),
            vc_buf_size=arch_dict.get('vc_buf_size', 8),
            flit_size=arch_dict.get('flit_size', 64),
            routing_delay=arch_dict.get('routing_delay', 0),
            vc_alloc_delay=arch_dict.get('vc_alloc_delay', 1),
            sw_alloc_delay=arch_dict.get('sw_alloc_delay', 1),
            st_final_delay=arch_dict.get('st_final_delay', 1),
            reconf_cycles=arch_dict.get('reconf_cycles', 2.0),
            ANY_comp_cycles=arch_dict.get('ANY_comp_cycles', 1.0),
            logger=arch_dict.get('logger', 0)
        )


@dataclass
class WorkloadEntry:
    """Single workload entry parsed from JSON"""
    id: int
    type: str
    size: int
    dependencies: List[int]
    
    # Fields for WRITE/WRITE_REQ
    src: int = None
    dst: int = None
    cl: int = None
    pt_required: int = None
    
    # Fields for COMP_OP
    layer_id: int = None
    weight_size: int = None
    input_range: Dict = None
    output_range: Dict = None
    ct_required: int = None
    node: int = None
    
    @classmethod
    def from_dict(cls, workload_dict: Dict) -> 'WorkloadEntry':
        return cls(
            id=workload_dict['id'],
            type=workload_dict['type'],
            size=workload_dict['size'],
            dependencies=workload_dict.get('dep', []),
            src=workload_dict.get('src'),
            dst=workload_dict.get('dst'),
            cl=workload_dict.get('cl'),
            pt_required=workload_dict.get('pt_required'),
            layer_id=workload_dict.get('layer_id'),
            weight_size=workload_dict.get('weight_size'),
            input_range=workload_dict.get('input_range'),
            output_range=workload_dict.get('output_range'),
            ct_required=workload_dict.get('ct_required'),
            node=workload_dict.get('node')
        )


class FastNoCSimulator:
    """Fast analytical NoC simulator replacement"""
    
    def __init__(self):
        self.topology_graph = None
        
    def parse_config(self, json_path: str) -> Tuple[ArchConfig, List[WorkloadEntry]]:
        """Parse JSON configuration file"""
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Parse architecture
        arch = ArchConfig.from_dict(config['arch'])
        
        # Parse workload
        workload = [WorkloadEntry.from_dict(w) for w in config['workload']]
        
        return arch, workload
    
    def build_topology_graph(self, arch: ArchConfig) -> nx.Graph:
        """Build NetworkX graph for the topology - cached for speed"""
        if self.topology_graph is not None:
            return self.topology_graph
            
        if arch.topology == "torus":
            # Create k-dimensional torus
            total_nodes = arch.n ** arch.k
            G = nx.Graph()
            G.add_nodes_from(range(total_nodes))
            
            # Add edges for torus topology
            for node in range(total_nodes):
                coords = self._node_to_coords(node, arch.n, arch.k)
                
                # Connect to neighbors in each dimension
                for dim in range(arch.k):
                    # Forward neighbor
                    next_coords = coords.copy()
                    next_coords[dim] = (coords[dim] + 1) % arch.n
                    next_node = self._coords_to_node(next_coords, arch.n)
                    G.add_edge(node, next_node)
                    
                    # Backward neighbor  
                    prev_coords = coords.copy()
                    prev_coords[dim] = (coords[dim] - 1) % arch.n
                    prev_node = self._coords_to_node(prev_coords, arch.n)
                    G.add_edge(node, prev_node)
            
            self.topology_graph = G
            return G
        
        elif arch.topology == "mesh":
            # Create k-dimensional mesh (no wraparound)
            total_nodes = arch.n ** arch.k
            G = nx.Graph()
            G.add_nodes_from(range(total_nodes))
            
            # Add edges for mesh topology
            for node in range(total_nodes):
                coords = self._node_to_coords(node, arch.n, arch.k)
                
                # Connect to neighbors in each dimension (without wraparound)
                for dim in range(arch.k):
                    # Forward neighbor (if not at edge)
                    if coords[dim] < arch.n - 1:
                        next_coords = coords.copy()
                        next_coords[dim] = coords[dim] + 1
                        next_node = self._coords_to_node(next_coords, arch.n)
                        G.add_edge(node, next_node)
                    
                    # Backward neighbor (if not at edge)
                    if coords[dim] > 0:
                        prev_coords = coords.copy()
                        prev_coords[dim] = coords[dim] - 1
                        prev_node = self._coords_to_node(prev_coords, arch.n)
                        G.add_edge(node, prev_node)
            
            self.topology_graph = G
            return G
        else:
            raise NotImplementedError(f"Topology {arch.topology} not implemented")
    
    def _node_to_coords(self, node: int, n: int, k: int) -> List[int]:
        """Convert node ID to k-dimensional coordinates"""
        coords = []
        for _ in range(k):
            coords.append(node % n)
            node //= n
        return coords
    
    def _coords_to_node(self, coords: List[int], n: int) -> int:
        """Convert k-dimensional coordinates to node ID"""
        node = 0
        multiplier = 1
        for coord in coords:
            node += coord * multiplier
            multiplier *= n
        return node
    
    def calculate_hop_distance(self, src: int, dst: int, arch: ArchConfig) -> int:
        """Calculate hop distance between two nodes"""
        if arch.topology == "torus":
            src_coords = self._node_to_coords(src, arch.n, arch.k)
            dst_coords = self._node_to_coords(dst, arch.n, arch.k)
            
            total_hops = 0
            for dim in range(arch.k):
                # Calculate minimum distance in torus (considering wraparound)
                direct_dist = abs(dst_coords[dim] - src_coords[dim])
                wrap_dist = arch.n - direct_dist
                total_hops += min(direct_dist, wrap_dist)
            
            return total_hops
        
        elif arch.topology == "mesh":
            src_coords = self._node_to_coords(src, arch.n, arch.k)
            dst_coords = self._node_to_coords(dst, arch.n, arch.k)
            
            total_hops = 0
            for dim in range(arch.k):
                # Calculate Manhattan distance in mesh (no wraparound)
                total_hops += abs(dst_coords[dim] - src_coords[dim])
            
            return total_hops
            
        else: #only if other topologies are added
            # Fallback to NetworkX shortest path
            G = self.build_topology_graph(arch)
            return nx.shortest_path_length(G, src, dst)
    
    def build_dependency_graph(self, workload: List[WorkloadEntry]) -> nx.DiGraph:
        """Build dependency graph for scheduling"""
        G = nx.DiGraph()
        
        # Add all tasks
        for task in workload:
            G.add_node(task.id, task=task)
        
        # Add dependencies
        for task in workload:
            for dep_id in task.dependencies:
                if dep_id != -1:  # -1 means no dependency
                    G.add_edge(dep_id, task.id)
        
        return G
    
    def calculate_message_latency(self, src: int, dst: int, size: int, arch: ArchConfig, is_reply: bool = False) -> float:
        """Calculate latency for a single message using your analytical formula"""
        # Network communication using your analytical model
        n_routers = self.calculate_hop_distance(src, dst, arch)
        
        # Calculate number of flits
        n_flits = 1 if is_reply else (size + arch.flit_size - 1) // arch.flit_size
        
        # Calculate number of ports (assuming torus topology)
        num_ports = 2 * arch.k  # bidirectional links in k dimensions
        
        # Parallelization factor k (adapt to ports and VCs)
        #paralelisation parameter can casue a problem here????
        k_parallel = min(4, max(2, min(num_ports, arch.num_vcs) // 2))
        
        # Link delay components
        width_flit = arch.flit_size  # bits
        width_phit = arch.width_phit  # physical link width
        
        if arch.topology == "torus":
            t_phy = 2  # physical delay
        else:
            t_phy = 1
            
        t_link = max(1, (width_flit + width_phit - 1) // width_phit) * t_phy
        
        # Calculate T_head^(hop) - time for head flit per hop
        # T_head^(hop) = 1 + 0.2 × adaptive_enabled + log2(num_vcs)/k + log2(num_ports^2)/k + 1
        # With adaptive_enabled = 0:
        t_head_hop = (1 + 
                        0.2 * 0 +  # adaptive_enabled = 0
                    math.log2(arch.num_vcs) / k_parallel +
                    math.log2(num_ports**2) / k_parallel + 
                    1)
        
        # Calculate T_body^(hop) - time for body flit per hop
        # T_body^(hop) = log2(num_ports^2)/k + 1
        t_body_hop = math.log2(num_ports**2) / k_parallel + 1
        
        # Your analytical formula:
        # T_packet = N_routers × (T_head^(hop) + T_link) + (N_flits-1) × max(T_body^(hop), T_link) + Queuing_delay
        
        head_term = n_routers * (t_head_hop + t_link)
        body_term = (n_flits - 1) * max(t_body_hop, t_link)
        
        # Queuing delay - simplified model
        # Simple queuing delay model - can be enhanced with traffic analysis
        # For now, assume minimal queuing delay proportional to path length and buffer constraints
        # Base queuing delay per router based on buffer size and VCs
        # Scale by number of routers in path
        # Add small random component for realism (deterministic but varies by message size)
        base_queuing_per_router = max(0, (1 / arch.vc_buf_size) * (1 / arch.num_vcs))
        total_queuing = n_routers * base_queuing_per_router
        size_factor = (size % 100) / 1000.0  # 0-0.1 range
        queuing_delay = total_queuing + size_factor
        
        total_latency = head_term + body_term + queuing_delay
        
        return int(round(total_latency))
    
    def _group_comp_ops_by_layer_and_node(self, workload: List[WorkloadEntry]) -> Dict:
        """Group COMP_OP operations by (node, layer_id) for batching"""
        groups = defaultdict(list)
        for task in workload:
            if task.type == 'COMP_OP' and task.layer_id is not None:
                groups[(task.node, task.layer_id)].append(task)
        return groups
    
    def _group_writes_by_dependencies(self, workload: List[WorkloadEntry], dep_graph) -> Dict:
        """Group WRITE operations - currently disabled as each WRITE generates individual REPLY"""
        # According to Rule 4, each WRITE generates its own REPLY, so no batching for now
        return defaultdict(list)
    
    def _is_task_in_batch(self, task: WorkloadEntry, comp_op_groups: Dict, write_groups: Dict, completed_tasks: Dict) -> bool:
        """Check if a task is part of a batch that should be processed together"""
        if task.type == 'COMP_OP' and task.layer_id is not None:
            group_key = (task.node, task.layer_id)
            if group_key in comp_op_groups and len(comp_op_groups[group_key]) > 1:
                # Sort batch by ID to get the correct first task
                batch = sorted(comp_op_groups[group_key], key=lambda t: t.id)
                first_task_id = batch[0].id
                
                # Skip if this is NOT the first task in the batch (it will be processed with the first)
                return task.id != first_task_id
        return False
    
    def _calculate_earliest_start_time(self, task: WorkloadEntry, task_completion_times: Dict, task_processing_completion_times: Dict) -> int:
        """Calculate the earliest start time for a task based on dependencies"""
        earliest_start = 0
        for dep_id in task.dependencies:
            if dep_id != -1 and dep_id in task_completion_times:
                # For COMP_OP tasks, they can start as soon as the WRITE processing finishes
                if task.type == 'COMP_OP' and dep_id in task_processing_completion_times:
                    earliest_start = max(earliest_start, task_processing_completion_times[dep_id])
                else:
                    earliest_start = max(earliest_start, task_completion_times[dep_id])
        return earliest_start
    
    def _process_write_operation(self, task: WorkloadEntry, earliest_start: int, arch: ArchConfig, 
                                node_send_available_times: Dict, node_compute_available_times: Dict,
                                task_completion_times: Dict, task_processing_completion_times: Dict,
                                task_start_times: Dict, write_groups: Dict, event_tracker: Optional[Any] = None):
        """Process WRITE/WRITE_REQ operations following the 7 rules"""
        
        # Rule 7: WRITE with dependency -1 does NOT need processing - data already at destination
        if task.dependencies == [-1]:
            # Data is already residing at destination, no network transfer or processing needed
            task_start_times[task.id] = earliest_start
            task_completion_times[task.id] = earliest_start  # Completes immediately
            task_processing_completion_times[task.id] = earliest_start  # Processing complete immediately
            
            # Track as instantaneous operation for visualization
            if event_tracker:
                event_tracker.track_write_process_start(task.id, task.dst, earliest_start, 0)
                event_tracker.track_write_process_end(task.id, task.dst, earliest_start)
            
            # No impact on node availability - data is already there
            return
        
        # Rule 6: Node can start sending another message after receiving REPLY from previous message
        earliest_start = max(earliest_start, node_send_available_times[task.src])
        task_start_times[task.id] = earliest_start
        
        if task.src == task.dst:
            # Local write - no network latency, just processing time
            processing_time = task.pt_required * 8 / arch.flit_size if task.pt_required else 0
            
            # Track local processing for visualization
            if event_tracker:
                event_tracker.track_write_process_start(task.id, task.dst, earliest_start, task.pt_required)
                event_tracker.track_write_process_end(task.id, task.dst, earliest_start + processing_time)
            
            # Rule 5: Node can start COMP_OP after receiving WRITE message (local processing completed)
            node_compute_available_times[task.dst] = earliest_start + processing_time
            node_send_available_times[task.src] = earliest_start + processing_time  # Can send next message immediately
            
            task_completion_times[task.id] = earliest_start + processing_time
            task_processing_completion_times[task.id] = earliest_start + processing_time
            return
        
        # Network communication case
        # Calculate timings
        network_latency = self.calculate_message_latency(task.src, task.dst, task.size, arch)
        processing_time = task.pt_required * 8 / arch.flit_size if task.pt_required else 0
        # Rule 4: Each WRITE generates REPLY message of one flit size
        reply_latency = self.calculate_message_latency(task.dst, task.src, 0, arch, is_reply=True)
        
        # Timeline: send -> arrive & process -> reply -> reply arrives
        send_start = earliest_start
        send_end = send_start + network_latency  # Message arrives at destination
        process_start = send_end
        process_end = process_start + processing_time
        reply_start = process_end
        reply_end = reply_start + reply_latency  # Reply arrives back at source
        
        # Track events for visualization
        if event_tracker:
            event_tracker.track_write_send_start(task.id, task.src, task.dst, send_start, task.size)
            event_tracker.track_write_send_end(task.id, task.src, task.dst, send_end)
            event_tracker.track_write_receive_start(task.id, task.src, task.dst, send_start)
            event_tracker.track_write_process_start(task.id, task.dst, process_start, task.pt_required)
            event_tracker.track_write_process_end(task.id, task.dst, process_end)
            event_tracker.track_reply_send_start(task.id, task.dst, task.src, reply_start)
            event_tracker.track_reply_receive_end(task.id, task.dst, task.src, reply_end)
        
        # Update node availability according to rules
        # Rule 6: Source node can't send next message until REPLY is received
        node_send_available_times[task.src] = reply_end
        # Rule 5: Destination node can start COMP_OP after processing WRITE message  
        node_compute_available_times[task.dst] = process_end
        
        task_completion_times[task.id] = reply_end  # Task complete when REPLY received
        task_processing_completion_times[task.id] = process_end  # Processing complete when WRITE processed
    
    def _process_comp_op_batch(self, task: WorkloadEntry, earliest_start: int, arch: ArchConfig,
                                node_compute_available_times: Dict, task_completion_times: Dict,
                                task_processing_completion_times: Dict, task_start_times: Dict,
                                comp_op_groups: Dict, layer_completion_times: Dict, event_tracker: Optional[Any] = None):
        """Process COMP_OP operations with layer-aware batching following the 6 rules"""
        group_key = (task.node, task.layer_id)
        
        if group_key in comp_op_groups and len(comp_op_groups[group_key]) > 1:
            # Rule 3: COMP_OP from same layer on same node can be done sequentially without messages
            batch = comp_op_groups[group_key]
            # Sort batch by ID to ensure smaller ID comes first (Rule 3)
            batch = sorted(batch, key=lambda t: t.id)
            
            # Rule 1: COMP_OP is processed by assigned node, node can't send but can receive during processing
            # Rule 5: Node can start COMP_OP after receiving WRITE message
            batch_start_time = max(earliest_start, node_compute_available_times[task.node])
            
            # Execute all operations in the batch sequentially (Rule 3: no messages between same layer ops)
            current_time = batch_start_time
            for batch_task in batch:
                task_start_times[batch_task.id] = current_time
                task_latency = batch_task.ct_required * arch.ANY_comp_cycles if batch_task.ct_required else 1
                current_time += task_latency
                
                task_completion_times[batch_task.id] = current_time
                task_processing_completion_times[batch_task.id] = current_time
            
            # Rule 1: During batch processing, node can't send messages but can receive
            # Update when the compute node can start next computation (after entire batch)
            node_compute_available_times[task.node] = current_time
            layer_completion_times[(task.node, task.layer_id)] = current_time
            
            # Track the batch execution for visualization
            if event_tracker:
                task_ids = [t.id for t in batch]
                event_tracker.track_batch_comp_ops(task_ids, task.node, batch_start_time, current_time, task.layer_id)
        else:
            # Process individual COMP_OP
            # Rule 5: Node can start COMP_OP after receiving WRITE message
            earliest_start = max(earliest_start, node_compute_available_times[task.node])
            task_start_times[task.id] = earliest_start
            
            task_latency = task.ct_required * arch.ANY_comp_cycles if task.ct_required else 1
            
            # Track individual COMP_OP for visualization
            if event_tracker:
                event_tracker.track_comp_op_start(task.id, task.node, earliest_start, task.ct_required)
                event_tracker.track_comp_op_end(task.id, task.node, earliest_start + task_latency)
            
            # Rule 1: During processing, node can't send messages but can receive
            node_compute_available_times[task.node] = earliest_start + task_latency
            task_completion_times[task.id] = earliest_start + task_latency
            task_processing_completion_times[task.id] = earliest_start + task_latency
        

    def simulate_execution(self, workload: List[WorkloadEntry], arch: ArchConfig, event_tracker: Optional[Any] = None) -> int:
        """Simulate execution with layer-aware batching optimization"""
        # Build dependency graph
        dep_graph = self.build_dependency_graph(workload)
        
        try:
            schedule_order = list(nx.topological_sort(dep_graph))
        except nx.NetworkXError:
            raise ValueError("Circular dependency detected in workload")
        
        # Group operations for layer-aware batching
        comp_op_groups = self._group_comp_ops_by_layer_and_node(workload)
        write_groups = self._group_writes_by_dependencies(workload, dep_graph)
        
        # Calculate execution times
        task_completion_times = {}
        task_start_times = {}
        task_processing_completion_times = {}  # When WRITE processing finishes (before REPLY)
        
        # Separate tracking for different resource types
        node_send_available_times = defaultdict(int)  # When nodes can send next packet
        node_compute_available_times = defaultdict(int)  # When nodes can start next computation
        
        # Track layer completion times for batched writes
        layer_completion_times = defaultdict(int)

        for task_id in schedule_order:
            task = next(w for w in workload if w.id == task_id)
            
            # Skip if this task is part of a batch that will be processed together
            if self._is_task_in_batch(task, comp_op_groups, write_groups, task_completion_times):
                continue
                
            # Calculate earliest start time based on dependencies
            earliest_start = self._calculate_earliest_start_time(task, task_completion_times, task_processing_completion_times)
            
            if task.type in ['WRITE', 'WRITE_REQ']:
                self._process_write_operation(task, earliest_start, arch, node_send_available_times, 
                                            node_compute_available_times, task_completion_times, 
                                            task_processing_completion_times, task_start_times, write_groups, event_tracker)
                
            elif task.type == 'COMP_OP':
                self._process_comp_op_batch(task, earliest_start, arch, node_compute_available_times, 
                                            task_completion_times, task_processing_completion_times, 
                                            task_start_times, comp_op_groups, layer_completion_times, event_tracker)
                        
        # Return the maximum completion time
        return int(np.ceil(max(task_completion_times.values()))) if task_completion_times else 0
    
    def run_simulation(self, json_path: str, dwrap: bool = True) -> Tuple[int, None]:
        """Main simulation entry point - matches original interface"""
        arch, workload = self.parse_config(json_path)
        total_latency = self.simulate_execution(workload, arch)
        
        # Return result and None for logger (as requested)
        return total_latency, None

class SimulatorStubAnalyticalModel:
    """Drop-in replacement maintaining the same interface"""
    
    def __init__(self):
        self.fast_sim = FastNoCSimulator()
    
    def run_simulation(self, json_path: str, dwrap: bool = True) -> Tuple[int, None]:
        """Maintains exact same interface as original"""
        return self.fast_sim.run_simulation(json_path, dwrap)


#Example usage:
#if __name__ == "__main__":
#    # Your existing code works unchanged:
#    stub = SimulatorStubAnalyticalModel()
#    start_time = time.time()
#    result, logger = stub.run_simulation("./data/partitioner_data/mapping.json", dwrap=True)
#    end_time = time.time()
#    elapsed_time = end_time - start_time
#    print(f"Total latency: {result} cycles")
#    print(f"Simulation time: {elapsed_time:.4f} seconds")