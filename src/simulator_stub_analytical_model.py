import json
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math

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
    output_delay: int
    credit_delay: int
    routing_delay: int
    vc_alloc_delay: int
    sw_alloc_delay: int
    st_prepare_delay: int
    st_final_delay: int
    internal_speedup: float
    input_speedup: int
    output_speedup: int
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
            output_delay=arch_dict.get('output_delay', 0),
            credit_delay=arch_dict.get('credit_delay', 0),
            routing_delay=arch_dict.get('routing_delay', 1),
            vc_alloc_delay=arch_dict.get('vc_alloc_delay', 1),
            sw_alloc_delay=arch_dict.get('sw_alloc_delay', 1),
            st_prepare_delay=arch_dict.get('st_prepare_delay', 1),
            st_final_delay=arch_dict.get('st_final_delay', 1),
            internal_speedup=arch_dict.get('internal_speedup', 1.0),
            input_speedup=arch_dict.get('input_speedup', 1),
            output_speedup=arch_dict.get('output_speedup', 1),
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
        """Build NetworkX graph for the topology - cached per instance for multiprocessing safety"""
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
    
    def calculate_message_latency(self, src: int, dst: int, size: int, arch: ArchConfig, is_reply: bool = False) -> float:
        """Calculate latency for a single message using your analytical formula"""
        # hop distance based on topology
        n_routers = self.calculate_hop_distance(src, dst, arch)
        
        # Calculate number of flits of a message
        n_flits = 1 if is_reply else (size + arch.flit_size - 1) // arch.flit_size
        
        t_link = 1 #I assume that injection time and ejection time to the router is also 1 cycle! 
        
        t_head_hop = arch.routing_delay + arch.vc_alloc_delay + arch.sw_alloc_delay + arch.st_prepare_delay + arch.st_final_delay #+ arch.output_delay + arch.credit_delay
        
        t_body_hop = arch.sw_alloc_delay + arch.st_prepare_delay + arch.st_final_delay
        
        queuing_delay = 0
        
        #head + body term
        T_packet =  n_routers * (t_head_hop + t_link + queuing_delay) + (n_flits - 1) * max(t_body_hop, t_link) 
        
        # Calculate number of ports (assuming torus topology)
        # (because + 1 for PE)
        #num_ports = 2 * arch.k + 1  # bidirectional links in k dimensions
        
        # Parallelization factor k (adapt to ports and VCs)
        #k_parallel = 4 #min(4, max(2, min(num_ports, arch.num_vcs) // 2))
        
        # Link delay components
        #width_flit = arch.flit_size  # bits
        #width_phit = arch.width_phit  # physical link width
        
        #it should be 1 flit -> 1 cycle in 1 channel
        #if arch.topology == "torus":
        #    t_phy = 2  # physical delay
        #else:
        #    t_phy = 1
            
        
        # Calculate T_head^(hop) - time for head flit per hop
        # T_head^(hop) = 1 + 0.2 × adaptive_enabled + log2(num_vcs)/k + log2(num_ports^2)/k + 1
        # With adaptive_enabled = 0:
        
        
        #analytical formula:
        # T_packet = N_routers × (T_head^(hop) + T_link) + (N_flits-1) × max(T_body^(hop), T_link) + Queuing_delay
        
        
        # Queuing delay - simplified model
        #base_queuing_per_router = max(0, (1 / arch.vc_buf_size) * (1 / arch.num_vcs))
        #total_queuing = n_routers * base_queuing_per_router
        #size_factor = (size % 100) / 1000.0  # 0-0.1 range
        #queuing_delay = 0 #total_queuing + size_factor
        
        #total_latency = head_term + body_term + queuing_delay
        
        
        #paper: Analytical latency model for networks-on-chip:
        # latency_packet = latency_head + latency_body
        # latency_body = (n_flits - 1) * max(T_body_hop, T_link)
        # latency_head = t_inj + n_hops * (t_routing + Waiting_time_i + t_switching) + (n_flits - 1) * t_link + t_eject
        return int(round(T_packet))
    
    def simulate_execution(self, workload: List[WorkloadEntry], arch: ArchConfig, tracker=None) -> int:
        """Enhanced execution simulation with proper dependency and parallelism handling"""
        
        task_completion_times = {}  # task_id -> completion_time
        node_available_times = {}   # node -> when it can start next operation
        
        # Initialize all nodes as available at time 0
        all_nodes = set()
        for task in workload:
            if task.node is not None:
                all_nodes.add(task.node)
            if task.src is not None:
                all_nodes.add(task.src)
            if task.dst is not None:
                all_nodes.add(task.dst)
        
        for node in all_nodes:
            node_available_times[node] = 0
        
        # Process tasks in dependency order
        completed_tasks = set()
        remaining_tasks = workload.copy()
        
        while remaining_tasks:
            # Find tasks that can be executed (all dependencies satisfied)
            ready_tasks = []
            for task in remaining_tasks:
                # Check if all dependencies are satisfied
                deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id != -1 and dep_id not in completed_tasks:
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # This might indicate a circular dependency or error in the workload
                print("WARNING: No ready tasks but still have remaining tasks")
                break
            
            # Process all ready tasks
            for task in ready_tasks:
                if task.type == "COMP_OP":
                    #print(f"DEBUG: Processing COMP_OP task {task.id}")
                    self._execute_single_comp_task(task, arch, node_available_times, task_completion_times, tracker)
                elif task.type in ["WRITE", "WRITE_REQ"]:
                    #print(f"DEBUG: Processing WRITE task {task.id}")
                    self._execute_single_write_task(task, arch, node_available_times, task_completion_times, tracker)
                
                completed_tasks.add(task.id)
                remaining_tasks.remove(task)
        
        # Return total execution time (latest completion)
        return max(task_completion_times.values()) if task_completion_times else 0
    
    def _execute_single_comp_task(self, task: WorkloadEntry, arch: ArchConfig, node_available_times: dict, 
                                task_completion_times: dict, tracker=None):
        """Execute a single COMP_OP task"""
        #print(f"DEBUG: _execute_single_comp_task called for task {task.id}, node {task.node}, tracker={tracker is not None}")
        # Calculate when this task can start
        earliest_start = node_available_times.get(task.node, 0)
        
        # Consider dependencies - allow more parallelism for COMP_OP
        for dep_id in task.dependencies:
            if dep_id != -1 and dep_id in task_completion_times:
                # For COMP_OP, we can start as soon as input data is available
                # Don't wait for the entire dependency chain to complete
                dep_completion = task_completion_times[dep_id]
                # Reduce dependency wait time for better pipeline parallelism
                earliest_start = max(earliest_start, dep_completion * 0.8)
        
        # COMP_OP tasks start at time 0 minimum (to match debug file behavior)
        earliest_start = max(earliest_start, 0)
        
        # Execute the computation task - use integer rounding
        task_latency = int(task.ct_required * arch.ANY_comp_cycles) if task.ct_required else 1
        completion_time = earliest_start + task_latency
        
        # Add size-proportional break time (like the real NoC simulator)
        # Break time proportional to computation size with a factor
        #break_factor = 0.0  # Adjust this factor as needed
        #break_time = max(1, int(task.ct_required * break_factor)) if task.ct_required else 1
        #completion_time += break_time
        
        # Track computation operation
        if tracker:
            tracker.track_comp_op_start(task.id, task.node, earliest_start, task.ct_required)
            tracker.track_comp_op_end(task.id, task.node, completion_time)
        
        # Update completion time and node availability
        task_completion_times[task.id] = completion_time
        node_available_times[task.node] = max(node_available_times[task.node], completion_time)
    
    def _execute_single_write_task(self, task: WorkloadEntry, arch: ArchConfig, node_available_times: dict, 
                                task_completion_times: dict, tracker=None):
        """Execute a single WRITE task"""
        # Calculate when this task can start - only consider source node availability, not destination
        earliest_start = node_available_times.get(task.src, 0)
        
        # New logic: WRITE can start immediately after COMP_OP completion
        dependency_ready_time = 0
        for dep_id in task.dependencies:
            if dep_id != -1 and dep_id in task_completion_times:
                # WRITE starts immediately when dependency (usually COMP_OP) completes
                dep_ready_time = task_completion_times[dep_id]
                dependency_ready_time = max(dependency_ready_time, dep_ready_time)
        
        earliest_start = max(earliest_start, dependency_ready_time)
        
        # No additional break time - start immediately after dependency
        
        if task.src == task.dst:
            # Local write
            processing_time = task.pt_required if task.pt_required else 1
            completion_time = earliest_start + processing_time
            
            if tracker:
                tracker.track_write_process_start(task.id, task.dst, earliest_start, task.pt_required)
                tracker.track_write_process_end(task.id, task.dst, completion_time)
            
            # For local writes, update both src and dst node availability
            task_completion_times[task.id] = completion_time
            node_available_times[task.src] = max(node_available_times.get(task.src, 0), completion_time)
        else:
            # Network write
            write_latency = self.calculate_message_latency(task.src, task.dst, task.size, arch)
            arrival_time = earliest_start + write_latency
            
            if task.pt_required and task.pt_required > 0:
                processing_end = arrival_time + task.pt_required
            else:
                processing_end = arrival_time
            
            reply_latency = self.calculate_message_latency(task.dst, task.src, arch.flit_size, arch, is_reply=True)
            # Double the reply latency as requested - replies take 2x more time
            reply_latency *= 2
            completion_time = processing_end + reply_latency
            
            if tracker:
                tracker.track_write_send_start(task.id, task.src, task.dst, earliest_start, task.size)
                tracker.track_write_send_end(task.id, task.src, task.dst, arrival_time)
                if task.pt_required and task.pt_required > 0:
                    tracker.track_write_receive_start(task.id, task.src, task.dst, arrival_time)
                    tracker.track_write_receive_end(task.id, task.src, task.dst, processing_end)
                tracker.track_reply_send_start(task.id, task.dst, task.src, processing_end)
                tracker.track_reply_send_end(task.id, task.dst, task.src, completion_time)
            
            # Communication hiding: Node can continue with computation immediately
            task_completion_times[task.id] = completion_time
            # Source node is NOT blocked - can continue with next COMP_OP immediately
            # Communication happens in background
            node_available_times[task.src] = max(node_available_times.get(task.src, 0), earliest_start)
            
            # Destination node also minimal blocking - can overlap with other work
            if task.dst is not None:
                node_available_times[task.dst] = max(node_available_times.get(task.dst, 0), arrival_time)
    
    def _execute_single_task(self, task: WorkloadEntry, arch: ArchConfig, node_available_times: dict, 
                            task_completion_times: dict, tracker=None):
        """Execute a single task based on its type"""
        
        # Calculate when this task can start (after dependencies and node availability)
        earliest_start = node_available_times[task.node]
        
        # For tasks with dependencies, ensure they complete first
        for dep_id in task.dependencies:
            if dep_id != -1 and dep_id in task_completion_times:
                earliest_start = max(earliest_start, task_completion_times[dep_id])
        
        if task.type == "COMP_OP":
            # Computation task
            task_latency = task.ct_required * arch.ANY_comp_cycles if task.ct_required else 1
            completion_time = earliest_start + task_latency
            
            # Track computation operation
            if tracker:
                tracker.track_comp_op_start(task.id, task.node, earliest_start, task.ct_required)
                tracker.track_comp_op_end(task.id, task.node, completion_time)
                
        elif task.type in ["WRITE", "WRITE_REQ"]:
            # Communication task - simplified model
            if task.src == task.dst:
                # Local communication
                processing_time = task.pt_required * 8 / arch.flit_size if task.pt_required else 0
                completion_time = earliest_start + processing_time
                
                # Track local processing
                if tracker:
                    tracker.track_write_process_start(task.id, task.dst, earliest_start, task.pt_required)
                    tracker.track_write_process_end(task.id, task.dst, completion_time)
            else:
                # Network communication
                communication_latency = self.calculate_message_latency(task.src, task.dst, task.size, arch)
                completion_time = earliest_start + communication_latency
                
                # Track network communication
                if tracker:
                    tracker.track_write_send_start(task.id, task.src, task.dst, earliest_start, task.size)
                    tracker.track_write_send_end(task.id, task.src, task.dst, completion_time)
                    tracker.track_write_receive_start(task.id, task.src, task.dst, completion_time)
        else:
            # Unknown task type - default behavior
            completion_time = earliest_start + 1
        
        # Update completion time and node availability
        task_completion_times[task.id] = completion_time
        node_available_times[task.node] = completion_time
    
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
