"""
Latency and Energy Analysis for NoC Simulations
Analysis of BookSim2 logger data to extract timing information for packets and computations
Energy estimation based on workload characteristics and operations
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict


def analyze_logger_events(logger) -> Dict:
    """
    Analyze BookSim2 logger events to extract latency and timing information

    Parameters:
    -----------
    logger : BookSim2 EventLogger object
        Logger object containing simulation events

    Returns:
    --------
    Dict containing analysis results
    """
    if logger is None:
        print("Warning: Logger is None, no analysis possible")
        return {}

    # Extract events from logger
    events = logger.events

    if not events:
        print("Warning: No events found in logger")
        return {}

    # Convert events to list of dictionaries for easier processing
    event_data = []
    for event in events:
        event_data.append({
            'id': event.id,
            'type': event.type,
            'cycle': event.cycle,
            'additional_info': event.additional_info,
            'ctype': event.ctype
        })

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(event_data)

    print(f"Total events: {len(df)}")
    print(f"Event types: {df['type'].unique()}")

    # Analyze packet latencies using logger directly
    packet_latencies = analyze_packet_latencies(logger)

    # Analyze computation times using logger directly
    computation_times = analyze_computation_times(logger)

    # Print detailed timing analysis
    print_timing_analysis(df, packet_latencies, computation_times)

    # Get timing analysis as DataFrames
    packets_df, computations_df = get_timing_analysis_dataframes(df, packet_latencies, computation_times)
    
    df = analyze_parallel_execution_time(packets_df,computations_df, df['cycle'].max())
    
    print_parallel_execution_analysis(df)

    # Analyze energy consumption
    #energy_df, total_energy = estimate_energy_from_data({
        #  'packet_latencies': packet_latencies,
        #  'computation_times': computation_times
    #})

    # Print energy analysis
    #print_energy_analysis(energy_df, total_energy)

    # Calculate energy efficiency metrics
    #efficiency_metrics = analyze_energy_efficiency(energy_df, packet_latencies, total_energy)

    return packets_df, computations_df, df


def communication_type_to_string(ctype: int) -> str:
    """
    Convert communication type integer to readable string
    """
    comm_types = {
        0: "ANY",
        1: "READ_REQ",
        2: "WRITE_REQ",
        3: "READ_ACK",
        4: "WRITE_ACK",
        5: "READ",
        6: "WRITE"
    }
    return comm_types.get(ctype, f"UNKNOWN({ctype})")


def analyze_packet_latencies(logger) -> List[Dict]:
    """
    Analyze packet latencies using the actual logger events structure
    Based on ani_utils.py implementation
    """
    try:
        import nocsim  # Import nocsim module
    except ImportError:
        print("Warning: Could not import nocsim")
        return []

    packet_latencies = []

    # Process OUT_TRAFFIC events (which contain the complete packet journey)
    for event in logger.events:
        if event.type == nocsim.EventType.OUT_TRAFFIC:
            packet_id = event.additional_info
            communication_type = communication_type_to_string(event.ctype)
            start_cycle = event.cycle
            sending_node = event.info.source
            receiving_node = event.info.dest

            # Extract timing from history bits
            send_start = None
            send_end = None
            receive_start = None
            receive_end = None

            for history_bit in event.info.history:
                if history_bit.start >= start_cycle:
                    # Sending node events
                    if history_bit.rsource == history_bit.rsink == sending_node:
                        if send_start is None or history_bit.start < send_start:
                            send_start = history_bit.start
                            send_end = history_bit.end

                    # Receiving node events
                    elif history_bit.rsource == history_bit.rsink == receiving_node:
                        if receive_start is None or history_bit.start < receive_start:
                            receive_start = history_bit.start
                            receive_end = history_bit.end

            # Calculate overall packet latency
            if send_start is not None and receive_end is not None:
                total_latency = receive_end - send_start

                packet_latencies.append({
                    'packet_id': packet_id,
                    'start_cycle': send_start,
                    'end_cycle': receive_end,
                    'latency': total_latency,
                    'source_node': sending_node,
                    'dest_node': receiving_node,
                    'communication_type': communication_type,
                    'send_duration': send_end - send_start if send_end and send_start else None,
                    'receive_duration': receive_end - receive_start if receive_end and receive_start else None
                })
        else:
            continue  # Ignore other event types
        
    return packet_latencies


def analyze_computation_times(logger) -> List[Dict]:
    """
    Analyze computation times by matching START_COMPUTATION with END_COMPUTATION events
    Based on ani_utils.py implementation
    """
    try:
        import nocsim  # Import nocsim module
    except ImportError:
        print("Warning: Could not import nocsim")
        return []

    computation_times = []

    # Process START_COMPUTATION events and find matching END_COMPUTATION
    for event in logger.events:
        if event.type == nocsim.EventType.START_COMPUTATION:
            node = event.info.node
            start_cycle = event.cycle
            comp_id = event.id

            # Find matching END_COMPUTATION event
            end_event = None
            for e in logger.events:
                if (e.type == nocsim.EventType.END_COMPUTATION and
                    e.info.node == node and
                    e.cycle > start_cycle):  # Ensure valid duration
                    end_event = e
                    break

            if end_event:
                duration = end_event.cycle - start_cycle
                computation_times.append({
                    'computation_id': comp_id,
                    'node': node,
                    'start_cycle': start_cycle,
                    'end_cycle': end_event.cycle,
                    'duration': duration,
                    'computation_time': end_event.info.ctime if hasattr(end_event.info, 'ctime') else None
                })
            else:
                print(f"Warning: No END_COMPUTATION found for node {node} at cycle {start_cycle}")

    return computation_times


def get_timing_analysis_dataframes(df: pd.DataFrame, packet_latencies: List[Dict], computation_times: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get timing analysis as DataFrames for packets and computations

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        packets_df: DataFrame with packet timing information
        computations_df: DataFrame with computation timing information
    """
    # Create packets DataFrame
    if packet_latencies:
        packets_data = []
        for packet in packet_latencies:
            packets_data.append({
                'Packet ID': packet['packet_id'],
                'Type': packet['communication_type'],
                'Start': packet['start_cycle'],
                'End': packet['end_cycle'],
                'Latency': packet['latency']
            })
        packets_df = pd.DataFrame(packets_data)
    else:
        packets_df = pd.DataFrame(columns=['Packet ID', 'Type', 'Start', 'End', 'Latency'])

    # Create computations DataFrame
    if computation_times:
        computations_data = []
        for comp in computation_times:
            computations_data.append({
                'Comp ID': comp['computation_id'],
                'Start': comp['start_cycle'],
                'End': comp['end_cycle'],
                'Duration': comp['duration']
            })
        computations_df = pd.DataFrame(computations_data)
    else:
        computations_df = pd.DataFrame(columns=['Comp ID', 'Start', 'End', 'Duration'])

    return packets_df, computations_df


def analyze_parallel_execution_time(packets_df: pd.DataFrame, computations_df: pd.DataFrame, total_simulation_cycles: int) -> Dict:
    """
    Analyze actual time spent on computations and data flow considering parallel execution.

    Parameters:
    -----------
    packets_df : pd.DataFrame
        DataFrame with packet timing information (columns: Packet ID, Type, Start, End, Latency)
    computations_df : pd.DataFrame
        DataFrame with computation timing information (columns: Comp ID, Start, End, Duration)
        Note: Should include 'node' column if available for proper parallelism analysis
    total_simulation_cycles : int
        Total simulation time in cycles

    Returns:
    --------
    Dict with timing analysis considering parallel execution
    """

    # Create timeline arrays to track what's happening at each cycle
    comp_timeline = np.zeros(total_simulation_cycles + 1, dtype=bool)
    packet_timeline = np.zeros(total_simulation_cycles + 1, dtype=bool)

    # For computation parallelism analysis
    comp_parallel_count = np.zeros(total_simulation_cycles + 1, dtype=int)

    # Mark computation periods and count parallel computations
    for _, row in computations_df.iterrows():
        start, end = int(row['Start']), int(row['End'])
        comp_timeline[start:end+1] = True
        comp_parallel_count[start:end+1] += 1

    # Mark packet transmission periods
    for _, row in packets_df.iterrows():
        start, end = int(row['Start']), int(row['End'])
        packet_timeline[start:end+1] = True

    # Calculate actual occupied time
    comp_occupied_cycles = np.sum(comp_timeline)
    packet_occupied_cycles = np.sum(packet_timeline)

    # Calculate overlap between computations and packets
    overlap_cycles = np.sum(comp_timeline & packet_timeline)

    # Calculate total active cycles (computation OR packet activity)
    total_active_cycles = np.sum(comp_timeline | packet_timeline)

    # Calculate idle time
    idle_cycles = total_simulation_cycles - total_active_cycles

    # Analyze computation parallelism
    max_parallel_comps = np.max(comp_parallel_count)
    avg_parallel_comps = np.mean(comp_parallel_count[comp_timeline]) if comp_occupied_cycles > 0 else 0

    # Calculate serial vs parallel computation time
    serial_comp_cycles = np.sum(comp_parallel_count == 1)
    parallel_comp_cycles = np.sum(comp_parallel_count > 1)

    # Total computation work (sum of all individual computation durations)
    total_comp_work = computations_df['Duration'].sum()

    # Parallelism efficiency
    parallelism_efficiency = total_comp_work / comp_occupied_cycles if comp_occupied_cycles > 0 else 0

    # Calculate percentages
    comp_percentage = (comp_occupied_cycles / total_simulation_cycles) * 100
    packet_percentage = (packet_occupied_cycles / total_simulation_cycles) * 100
    overlap_percentage = (overlap_cycles / total_simulation_cycles) * 100
    active_percentage = (total_active_cycles / total_simulation_cycles) * 100
    idle_percentage = (idle_cycles / total_simulation_cycles) * 100
    serial_comp_percentage = (serial_comp_cycles / total_simulation_cycles) * 100
    parallel_comp_percentage = (parallel_comp_cycles / total_simulation_cycles) * 100

    return {
        'total_simulation_cycles': total_simulation_cycles,
        'computation_occupied_cycles': comp_occupied_cycles,
        'packet_occupied_cycles': packet_occupied_cycles,
        'overlap_cycles': overlap_cycles,
        'total_active_cycles': total_active_cycles,
        'idle_cycles': idle_cycles,
        'computation_percentage': comp_percentage,
        'packet_percentage': packet_percentage,
        'overlap_percentage': overlap_percentage,
        'active_percentage': active_percentage,
        'idle_percentage': idle_percentage,
        'computation_efficiency': comp_occupied_cycles / total_active_cycles * 100 if total_active_cycles > 0 else 0,
        'packet_efficiency': packet_occupied_cycles / total_active_cycles * 100 if total_active_cycles > 0 else 0,
        # Computation parallelism metrics
        'max_parallel_computations': max_parallel_comps,
        'avg_parallel_computations': avg_parallel_comps,
        'serial_computation_cycles': serial_comp_cycles,
        'parallel_computation_cycles': parallel_comp_cycles,
        'serial_comp_percentage': serial_comp_percentage,
        'parallel_comp_percentage': parallel_comp_percentage,
        'total_computation_work': total_comp_work,
        'parallelism_efficiency': parallelism_efficiency,
        'parallelism_speedup': parallelism_efficiency,  # Same as efficiency in this context
        'computation_utilization': avg_parallel_comps / max_parallel_comps * 100 if max_parallel_comps > 0 else 0
    }


def print_parallel_execution_analysis(analysis: Dict):
    """Print detailed parallel execution analysis"""
    print("\n" + "="*60)
    print("PARALLEL EXECUTION TIME ANALYSIS")
    print("="*60)

    print(f"\nTime Distribution:")
    print(f"Total simulation cycles: {analysis['total_simulation_cycles']}")
    print(f"Computation occupied time: {analysis['computation_occupied_cycles']} cycles ({analysis['computation_percentage']:.1f}%)")
    print(f"Packet transmission time: {analysis['packet_occupied_cycles']} cycles ({analysis['packet_percentage']:.1f}%)")
    print(f"Overlap (comp + packet) time: {analysis['overlap_cycles']} cycles ({analysis['overlap_percentage']:.1f}%)")
    print(f"Total active time: {analysis['total_active_cycles']} cycles ({analysis['active_percentage']:.1f}%)")
    print(f"Idle time: {analysis['idle_cycles']} cycles ({analysis['idle_percentage']:.1f}%)")

    print(f"\nComputation Parallelism Analysis:")
    print(f"Total computation work: {analysis['total_computation_work']} cycles (if serial)")
    print(f"Actual computation time: {analysis['computation_occupied_cycles']} cycles")
    print(f"Parallelism speedup: {analysis['parallelism_speedup']:.2f}x")
    print(f"Max parallel computations: {analysis['max_parallel_computations']}")
    print(f"Avg parallel computations: {analysis['avg_parallel_computations']:.2f}")
    print(f"Computation utilization: {analysis['computation_utilization']:.1f}%")

    print(f"\nComputation Time Breakdown:")
    print(f"Serial computation time: {analysis['serial_computation_cycles']} cycles ({analysis['serial_comp_percentage']:.1f}%)")
    print(f"Parallel computation time: {analysis['parallel_computation_cycles']} cycles ({analysis['parallel_comp_percentage']:.1f}%)")

    print(f"\nEfficiency Metrics:")
    print(f"Computation efficiency: {analysis['computation_efficiency']:.1f}% (of active time)")
    print(f"Packet efficiency: {analysis['packet_efficiency']:.1f}% (of active time)")

    if analysis['overlap_cycles'] > 0:
        print(f"\nComp-Packet Parallelism:")
        print(f"Computation + packet overlap: {analysis['overlap_cycles']} cycles")
        print(f"Overlap utilization: {analysis['overlap_percentage']:.1f}% of total time")


def print_timing_analysis(df: pd.DataFrame, packet_latencies: List[Dict], computation_times: List[Dict]):
    """
    Print detailed timing analysis
    """
    print("\n" + "="*60)
    print("TIMING ANALYSIS REPORT")
    print("="*60)

    # Overall simulation info
    print(f"\nSimulation Overview:")
    print(f"Total simulation cycles: {df['cycle'].max() if not df.empty else 0}")
    print(f"Total events: {len(df)}")

    # Event type breakdown
    print(f"\nEvent Type Breakdown:")
    if not df.empty:
        event_counts = df['type'].value_counts()
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")

    # Packet latency analysis
    print(f"\nPacket Latency Analysis:")
    print(f"Total packets analyzed: {len(packet_latencies)}")

    #if packet_latencies:
    #    latencies = [p['latency'] for p in packet_latencies]
    #    print(f"Average packet latency: {np.mean(latencies):.2f} cycles")
    #    print(f"Min packet latency: {np.min(latencies)} cycles")
    #    print(f"Max packet latency: {np.max(latencies)} cycles")
    #    print(f"Std dev packet latency: {np.std(latencies):.2f} cycles")

    #    print(f"\nDetailed Packet Timing:")
    #    print(f"{'Packet ID':<10} {'Type':<12} {'Start':<8} {'End':<8} {'Latency':<8}")
    #    print("-" * 40)
    #    for packet in sorted(packet_latencies, key=lambda x: x['start_cycle'])[:10]:  # Show first 10
    #        print(f"{packet['packet_id']:<10} {packet['communication_type']:<12} {packet['start_cycle']:<8} {packet['end_cycle']:<8} {packet['latency']:<8}")

    #    if len(packet_latencies) > 10:
    #        print(f"... and {len(packet_latencies) - 10} more packets")

    # Computation time analysis
    print(f"\nComputation Time Analysis:")
    print(f"Total computations analyzed: {len(computation_times)}")

    if computation_times:
        durations = [c['duration'] for c in computation_times]
        print(f"Average computation time: {np.mean(durations):.2f} cycles")
        print(f"Min computation time: {np.min(durations)} cycles")
        print(f"Max computation time: {np.max(durations)} cycles")
        print(f"Std dev computation time: {np.std(durations):.2f} cycles")

        #print(f"\nDetailed Computation Timing:")
        #print(f"{'Comp ID':<10} {'Start':<8} {'End':<8} {'Duration':<8}")
        #print("-" * 40)
        for comp in sorted(computation_times, key=lambda x: x['start_cycle'])[:10]:  # Show first 10
            print(f"{comp['computation_id']:<10} {comp['start_cycle']:<8} {comp['end_cycle']:<8} {comp['duration']:<8}")

        if len(computation_times) > 10:
            print(f"... and {len(computation_times) - 10} more computations")


# ============================================================================
# ENERGY ESTIMATION FUNCTIONS
# ============================================================================

# Data from Horowitz: Computing's Energy Problem
DEFAULT_ENERGY_PARAMS = {
    "COMP_OP":   {"energy_per_flop": 1.1e-12},   # 0.4 pJ per flop (example)
    "WRITE":     {"energy_per_byte": 5e-12},   # 5 pJ per byte (example)
    "WRITE_REQ": {"energy_per_byte": 5e-12},   # same as WRITE by default
    # added types for SRAM and reply
    "SRAM_READ":  {"energy_per_byte": 10.5e-12},  # example: 10.5 pJ/byte
    "SRAM_WRITE": {"energy_per_byte": 12.5e-12},  # example: 12.5 pJ/byte
    "REPLY":      {"energy_per_byte": 5e-12},  # treat reply as network byte cost by default
    # fallback DEFAULT if needed
    "DEFAULT":    {"energy_per_byte": 1.1e-12, "energy_per_flop": 0.4e-12}
}


def load_workload_json(config_path: str) -> Dict:
    """Load JSON configuration from file path."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON from '{config_path}': {e}")
        return {}


def estimate_energy_from_config(config_path: str, energy_params: Dict = None) -> Tuple[pd.DataFrame, float]:
    """
    Estimate energy consumption from workload configuration file.

    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file
    energy_params : Dict, optional
        Energy parameters for different operation types

    Returns:
    --------
    Tuple[pd.DataFrame, float]
        DataFrame with energy breakdown and total energy
    """
    if energy_params is None:
        energy_params = DEFAULT_ENERGY_PARAMS.copy()

    data = load_workload_json(config_path)
    return estimate_energy_from_data(data, energy_params)


def estimate_energy_from_data(data: Dict, energy_params: Dict = None) -> Tuple[pd.DataFrame, float]:
    """
    Estimate energy consumption from workload data or analysis results.

    Based on energy_estimator.py logic:
    - For each WRITE: adds SRAM write + REPLY packet
    - For each COMP_OP: adds SRAM read
    """
    if energy_params is None:
        energy_params = DEFAULT_ENERGY_PARAMS.copy()

    agg = defaultdict(lambda: {"count": 0, "total_bytes": 0, "total_flops": 0})

    # Check if this is analysis results (contains packet_latencies) or workload data
    if 'packet_latencies' in data and 'computation_times' in data:
        # This is analysis results from logger - convert to energy estimation format
        packet_latencies = data.get('packet_latencies', [])
        computation_times = data.get('computation_times', [])

        # Convert packets to WRITE operations (simplified assumption)
        for packet in packet_latencies:
            agg["WRITE"]["count"] += 1
            agg["WRITE"]["total_bytes"] += 64  # Assume 64 bytes per packet (flit size)

        # Convert computations to COMP_OP operations
        for comp in computation_times:
            agg["COMP_OP"]["count"] += 1
            agg["COMP_OP"]["total_flops"] += comp.get('duration', 1)  # Use duration as flops estimate

        reply_bytes = 8  # Default reply size

    else:
        # Original workload data format
        workload = data.get("workload", [])
        if not isinstance(workload, list):
            print("Input data does not contain a 'workload' array.")
            return pd.DataFrame(), 0.0

        # Determine flit size (bits) -> convert to bytes (ceiling)
        arch = data.get("arch", {}) or {}
        flit_bits = None
        # Try common keys
        for key in ("flit_size", "width_phit"):
            if key in arch:
                try:
                    flit_bits = int(arch[key])
                    break
                except Exception:
                    pass

        if flit_bits is None:
            print("Warning: flit size not found in arch. REPLY bytes set to 8.")
            reply_bytes = 8  # Default to 8 bytes (64 bits)
        else:
            reply_bytes = (flit_bits + 7) // 8  # ceil bits->bytes

        # Process workload entries
        for entry in workload:
            typ = entry.get("type", "UNKNOWN")
            agg[typ]["count"] += 1

            # Size field assumed bytes
            try:
                size_val = int(entry.get("size", 0) or 0)
            except Exception:
                size_val = 0
            agg[typ]["total_bytes"] += size_val

            # Sum ct_required and pt_required if present (flops)
            flops = 0
            for key in ("ct_required", "pt_required"):
                if key in entry and entry[key] is not None:
                    try:
                        flops += int(entry[key])
                    except Exception:
                        pass
            agg[typ]["total_flops"] += flops

            # Additional synthetic events per energy_estimator.py logic
            if typ == "WRITE":
                # Add a SRAM write of the same size
                agg["SRAM_WRITE"]["count"] += 1
                agg["SRAM_WRITE"]["total_bytes"] += size_val
                # Add a REPLY packet of size 1 flit (converted to bytes)
                if reply_bytes > 0:
                    agg["REPLY"]["count"] += 1
                    agg["REPLY"]["total_bytes"] += reply_bytes
            elif typ == "COMP_OP":
                # Add a SRAM read of the COMP_OP 'size'
                agg["SRAM_READ"]["count"] += 1
                agg["SRAM_READ"]["total_bytes"] += size_val

    # Apply energy augmentation for logger analysis results
    if 'packet_latencies' in data and 'computation_times' in data:
        # Add SRAM operations for WRITE traffic (simplified)
        write_count = agg["WRITE"]["count"]
        write_bytes = agg["WRITE"]["total_bytes"]

        if write_count > 0:
            agg["SRAM_WRITE"]["count"] += write_count
            agg["SRAM_WRITE"]["total_bytes"] += write_bytes
            agg["REPLY"]["count"] += write_count
            agg["REPLY"]["total_bytes"] += write_count * reply_bytes

        # Add SRAM read operations for computations
        comp_count = agg["COMP_OP"]["count"]
        comp_flops = agg["COMP_OP"]["total_flops"]

        if comp_count > 0:
            agg["SRAM_READ"]["count"] += comp_count
            agg["SRAM_READ"]["total_bytes"] += comp_flops  # Use flops as byte estimate

    # Compute energy
    rows = []
    total_energy = 0.0

    for typ, vals in agg.items():
        bytes_ = vals["total_bytes"]
        flops_ = vals["total_flops"]
        energy_from_bytes = 0.0
        energy_from_flops = 0.0

        params = energy_params.get(typ, None)
        if params is None:
            params = energy_params.get("DEFAULT", {})

        if "energy_per_byte" in params and bytes_:
            energy_from_bytes = bytes_ * params["energy_per_byte"]
        if "energy_per_flop" in params and flops_:
            energy_from_flops = flops_ * params["energy_per_flop"]

        # Extra fallback (if DEFAULT present)
        if energy_from_bytes == 0.0 and "energy_per_byte" in energy_params.get("DEFAULT", {}):
            energy_from_bytes = bytes_ * energy_params["DEFAULT"]["energy_per_byte"]
        if energy_from_flops == 0.0 and "energy_per_flop" in energy_params.get("DEFAULT", {}):
            energy_from_flops = flops_ * energy_params["DEFAULT"]["energy_per_flop"]

        energy = energy_from_bytes + energy_from_flops
        total_energy += energy

        rows.append({
            "type": typ,
            "count": vals["count"],
            "total_bytes": bytes_,
            "total_flops": flops_,
            "energy_bytes_J": energy_from_bytes,
            "energy_flops_J": energy_from_flops,
            "energy_total_J": energy
        })

    df = pd.DataFrame(rows).sort_values("type").reset_index(drop=True)
    return df, total_energy


def print_energy_analysis(energy_df: pd.DataFrame, total_energy: float):
    """Print detailed energy analysis report."""
    print("\n" + "="*60)
    print("ENERGY ANALYSIS REPORT")
    print("="*60)

    if energy_df.empty:
        print("No workload entries found for energy analysis.")
        return

    pd.set_option("display.float_format", "{:.6e}".format)
    print("\nEnergy Breakdown by Operation Type:")
    print(energy_df.to_string(index=False))

    print(f"\nTOTAL estimated energy = {total_energy:.6e} J  ({total_energy*1e6:.3f} microjoules)")

    # Additional summary statistics
    if not energy_df.empty:
        print(f"\nSummary Statistics:")
        print(f"Total operations: {energy_df['count'].sum()}")
        print(f"Total bytes processed: {energy_df['total_bytes'].sum()}")
        print(f"Total flops executed: {energy_df['total_flops'].sum()}")

        # Energy breakdown by category
        byte_energy = energy_df['energy_bytes_J'].sum()
        flop_energy = energy_df['energy_flops_J'].sum()
        print(f"Energy from data movement: {byte_energy:.6e} J ({byte_energy/total_energy*100:.1f}%)")
        print(f"Energy from computation: {flop_energy:.6e} J ({flop_energy/total_energy*100:.1f}%)")


def analyze_energy_efficiency(packet_latencies: List[Dict], energy_df: pd.DataFrame, total_energy: float) -> Dict:
    """
    Analyze energy efficiency metrics combining latency and energy data.

    Returns:
    --------
    Dict with energy efficiency metrics
    """
    if not packet_latencies or energy_df.empty:
        return {}

    # Calculate energy efficiency metrics
    avg_latency = np.mean([p['latency'] for p in packet_latencies])
    total_packets = len(packet_latencies)

    # Energy per packet
    energy_per_packet = total_energy / total_packets if total_packets > 0 else 0

    # Energy-delay product
    energy_delay_product = total_energy * avg_latency

    # Throughput (packets per cycle)
    max_cycle = max([p['end_cycle'] for p in packet_latencies]) if packet_latencies else 1
    throughput = total_packets / max_cycle if max_cycle > 0 else 0

    # Energy per bit (assuming some average packet size)
    total_bytes = energy_df['total_bytes'].sum()
    energy_per_bit = total_energy / (total_bytes * 8) if total_bytes > 0 else 0

    return {
        'total_energy_J': total_energy,
        'avg_latency_cycles': avg_latency,
        'energy_per_packet_J': energy_per_packet,
        'energy_delay_product': energy_delay_product,
        'throughput_packets_per_cycle': throughput,
        'energy_per_bit_J': energy_per_bit,
        'total_packets': total_packets,
        'total_cycles': max_cycle
    }


def parallel_analysis_to_dataframe(parallel_analysis: Dict, strategy_name: str = None) -> pd.DataFrame:
    """
    Convert parallel execution analysis dictionary to DataFrame for CSV export.

    Parameters:
    -----------
    parallel_analysis : Dict
        Result from analyze_parallel_execution_time()
    strategy_name : str, optional
        Name of the strategy/configuration being analyzed

    Returns:
    --------
    pd.DataFrame with parallel execution metrics
    """

    data = {
        'strategy': [strategy_name] if strategy_name else ['default'],
        'total_simulation_cycles': [parallel_analysis['total_simulation_cycles']],
        'computation_occupied_cycles': [parallel_analysis['computation_occupied_cycles']],
        'packet_occupied_cycles': [parallel_analysis['packet_occupied_cycles']],
        'overlap_cycles': [parallel_analysis['overlap_cycles']],
        'total_active_cycles': [parallel_analysis['total_active_cycles']],
        'idle_cycles': [parallel_analysis['idle_cycles']],
        'computation_percentage': [parallel_analysis['computation_percentage']],
        'packet_percentage': [parallel_analysis['packet_percentage']],
        'overlap_percentage': [parallel_analysis['overlap_percentage']],
        'active_percentage': [parallel_analysis['active_percentage']],
        'idle_percentage': [parallel_analysis['idle_percentage']],
        'computation_efficiency': [parallel_analysis['computation_efficiency']],
        'packet_efficiency': [parallel_analysis['packet_efficiency']],
        'max_parallel_computations': [parallel_analysis['max_parallel_computations']],
        'avg_parallel_computations': [parallel_analysis['avg_parallel_computations']],
        'serial_computation_cycles': [parallel_analysis['serial_computation_cycles']],
        'parallel_computation_cycles': [parallel_analysis['parallel_computation_cycles']],
        'serial_comp_percentage': [parallel_analysis['serial_comp_percentage']],
        'parallel_comp_percentage': [parallel_analysis['parallel_comp_percentage']],
        'total_computation_work': [parallel_analysis['total_computation_work']],
        'parallelism_efficiency': [parallel_analysis['parallelism_efficiency']],
        'parallelism_speedup': [parallel_analysis['parallelism_speedup']],
        'computation_utilization': [parallel_analysis['computation_utilization']]
    }

    return pd.DataFrame(data)


# Example usage function
def analyze_simulation_results(config_path: str, output_dir: str = ".", create_plots: bool = True):
    """
    Complete analysis pipeline for a simulation

    Parameters:
    -----------
    config_path : str
        Path to the simulation configuration file
    output_dir : str
        Directory to save analysis results
    create_plots : bool
        Whether to create visualization plots
    """
    import simulator_stub as ss

    # Run BookSim2 simulation
    print("Running BookSim2 simulation...")
    stub = ss.SimulatorStub()
    result, logger = stub.run_simulation(config_path, verbose=True)

    print(f"BookSim2 simulation completed: {result} cycles")

    # Analyze logger data (includes timing and energy analysis)
    analysis_results = analyze_logger_events(logger)

    # Create plots if requested
    if create_plots and analysis_results:
        plot_path = f"{output_dir}/timing_analysis.png"
        plot_timing_analysis(analysis_results, plot_path)

    # Print summary of results
    #print(f"\n=== ANALYSIS SUMMARY ===")
    #print(f"Total simulation time: {analysis_results.get('total_simulation_time', 0)} cycles")
    #print(f"Total packets analyzed: {len(analysis_results.get('packet_latencies', []))}")
    #print(f"Total computations analyzed: {len(analysis_results.get('computation_times', []))}")
    #print(f"Total energy consumption: {analysis_results.get('total_energy_J', 0):.6e} J")

    #if 'efficiency_metrics' in analysis_results:
    #    eff = analysis_results['efficiency_metrics']
    #    print(f"Energy per packet: {eff.get('energy_per_packet_J', 0):.6e} J")
    #    print(f"Energy-delay product: {eff.get('energy_delay_product', 0):.6e}")

    return analysis_results

