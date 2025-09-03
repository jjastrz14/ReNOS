import json
import networkx as nx
from typing import Dict, List, Tuple, Any
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
        

    def simulate_execution(self, workload: List[WorkloadEntry], arch: ArchConfig) -> int:
        """Simulate execution and return total latency"""
        # Build dependency graph
        dep_graph = self.build_dependency_graph(workload)
        
        try:
            schedule_order = list(nx.topological_sort(dep_graph))
        except nx.NetworkXError:
            raise ValueError("Circular dependency detected in workload")
        
        # Calculate execution times
        task_completion_times = {}
        task_start_times = {}
        node_available_times = defaultdict(int)  # Track when nodes can send next packet

        for task_id in schedule_order:
            task = next(w for w in workload if w.id == task_id)
            
            # Calculate earliest start time based on dependencies
            earliest_start = 0
            for dep_id in task.dependencies:
                if dep_id != -1 and dep_id in task_completion_times:
                    earliest_start = max(earliest_start, task_completion_times[dep_id])
            
            # For WRITE operations, also need to wait for node to be available
            if task.type in ['WRITE', 'WRITE_REQ']:
                earliest_start = max(earliest_start, node_available_times[task.src])
                
            task_start_times[task_id] = earliest_start
            
            if task.type in ['WRITE', 'WRITE_REQ']:
                
                if task.src == task.dst:
                    # Local write, no network latency
                    #below uncomment if you want to consider processing time for local writes
                    processing_time = 0 # task.pt_required * 8 / arch.flit_size if task.pt_required else 0
                    task_latency = processing_time
                    task_completion_times[task_id] = earliest_start + task_latency
                    continue
                
                # Calculate network latency for the packet
                network_latency = self.calculate_message_latency(task.src, task.dst, task.size, arch)
                
                # Calculate processing time at destination
                processing_time = task.pt_required * 8 / arch.flit_size if task.pt_required else 0
                
                # Calculate reply latency (small packet with only head)
                reply_latency = self.calculate_message_latency(task.dst, task.src, 0, arch, is_reply=True)
                
                # Total latency for this operation
                task_latency = network_latency + processing_time + reply_latency
                
                # Update when the source node can send next packet
                node_available_times[task.src] = earliest_start + task_latency
                
                task_completion_times[task_id] = earliest_start + task_latency
                
            elif task.type == 'COMP_OP':
                # Computation operation
                task_latency = task.ct_required * arch.ANY_comp_cycles if task.ct_required else 1
                task_completion_times[task_id] = earliest_start + task_latency
                    
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


# Example usage:
#if __name__ == "__main__":
#    # Your existing code works unchanged:
#    stub = SimulatorStubAnalytical_noc_model()
#    start_time = time.time()
#    result, logger = stub.run_simulation("./DATE26/best_solution.json", dwrap=True)
#    elapsed_time = time.time() - start_time
#    print(f"Total latency: {result} cycles")
#    print(f"Simulation time: {elapsed_time:.4f} seconds")