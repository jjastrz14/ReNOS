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


def analyze_logger_events(logger, config_data=None) -> Dict:
    """
    Analyze BookSim2 logger events to extract latency and timing information

    Parameters:
    -----------
    logger : BookSim2 EventLogger object
        Logger object containing simulation events
    config_data : dict or str, optional
        Configuration data (dict) or path to JSON config file (str).
        Used to extract 'k' parameter for number of nodes (num_nodes = k * k)

    Returns:
    --------
    Dict containing analysis results including updated energy parameters
    """
    if logger is None:
        print("Warning: Logger is None, no analysis possible")
        return {}

    # Load config data if it's a path
    if isinstance(config_data, str):
        config_data = load_workload_json(config_data)

    # Check for power summary and calculate NoC energy per byte
    print("\n" + "="*60)
    print("POWER SUMMARY CHECK")
    print("="*60)

    noc_energy_result = None
    updated_energy_params = None

    if logger.has_power_summary:
        ps = logger.power_summary
        print(f"✓ Power summary available!")
        print(f"  Total Power: {ps.total_power:.6f} W")
        print(f"  Clock Frequency: {ps.fclk:.3e} Hz")
        print(f"  Completion Time: {ps.completion_time_cycles} cycles")
        print(f"  Flit Width: {ps.flit_width_bits} bits")
        print(f"  Vdd: {ps.vdd} V")

        # Calculate total energy
        total_energy_J = ps.total_power * ps.completion_time_cycles / ps.fclk
        print(f"  Total Energy: {total_energy_J:.6e} J ({total_energy_J*1e6:.3f} µJ)")

        # Calculate actual NoC energy per byte if stats summary is also available
        if logger.has_stats_summary and packet_latencies:
            print("\n" + "="*60)
            print("NOC ENERGY CALCULATION (PER-PACKET METHOD)")
            print("="*60)
            try:
                # Use per-packet energy calculation (more accurate)
                noc_energy_result = calculate_noc_energy_per_byte_per_packet(logger, packet_latencies, config_data)
                updated_energy_params, _ = get_updated_energy_params(logger, packet_latencies=packet_latencies, config_data=config_data, use_per_packet=True)

                print(f"✓ Calculated per-packet NoC energy using {noc_energy_result['total_packets_analyzed']} packets")
                print(f"  Average packet size: {noc_energy_result['avg_packet_size_bytes']:.1f} bytes ({noc_energy_result['avg_packet_length_flits']:.1f} flits)")
                print(f"\n  Energy per Byte by Communication Type:")

                for comm_type in ['WRITE', 'WRITE_REQ', 'REPLY', 'READ', 'READ_REQ']:
                    count = noc_energy_result.get(f'{comm_type}_packet_count', 0)
                    if count > 0:
                        energy_pJ = noc_energy_result.get(f'{comm_type}_energy_per_byte_pJ', 0)
                        std_pJ = noc_energy_result.get(f'{comm_type}_energy_per_byte_std_pJ', 0)
                        print(f"    {comm_type:10s}: {energy_pJ:8.3f} ± {std_pJ:6.3f} pJ/byte  ({count} packets)")

                print(f"\n  Overall average: {noc_energy_result.get('overall_energy_per_byte_pJ', 0):.3f} pJ/byte")

                print(f"\n  Updated Energy Parameters:")
                print(f"    WRITE:     {updated_energy_params['WRITE']['energy_per_byte']*1e12:.3f} pJ/byte (from per-packet analysis)")
                print(f"    WRITE_REQ: {updated_energy_params['WRITE_REQ']['energy_per_byte']*1e12:.3f} pJ/byte (from per-packet analysis)")
                print(f"    REPLY:     {updated_energy_params['REPLY']['energy_per_byte']*1e12:.3f} pJ/byte (from per-packet analysis)")

                print(f"\n  Comparison with defaults:")
                default_noc = DEFAULT_ENERGY_PARAMS['WRITE']['energy_per_byte']
                actual_noc = updated_energy_params['WRITE']['energy_per_byte']
                ratio = actual_noc / default_noc
                print(f"    Default: {default_noc*1e12:.3f} pJ/byte")
                print(f"    Actual:  {actual_noc*1e12:.3f} pJ/byte")
                print(f"    Ratio:   {ratio:.2f}x")
            except Exception as e:
                print(f"✗ Could not calculate NoC energy per byte: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("✗ No power summary available")
        print("  Make sure 'sim_power': 1 is set in your config file")

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

    print(f"\nTotal events: {len(df)}")
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

    # Return results including updated energy params
    return {
        'packets_df': packets_df,
        'computations_df': computations_df,
        'parallel_analysis': df,
        'noc_energy': noc_energy_result,
        'updated_energy_params': updated_energy_params
    }


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


def calculate_noc_energy_per_byte_per_packet(logger, packet_latencies, config_data=None):
    """
    Calculate actual NoC energy per byte for each packet based on individual packet latencies.
    This is more accurate than the aggregate approach as it accounts for per-packet timing.

    Parameters:
    -----------
    logger : EventLogger
        Logger object with power_summary and stats_summary
    packet_latencies : List[Dict]
        List of packet latency dictionaries from analyze_packet_latencies()
    config_data : dict, optional
        Configuration data containing power and flit size info

    Returns:
    --------
    dict with per-packet-type energy metrics
    """
    if not logger.has_power_summary:
        raise ValueError("Logger does not have power summary. Set sim_power=1 in config.")

    if not logger.has_stats_summary:
        raise ValueError("Logger does not have stats summary. Set logger=1 in config.")

    ps = logger.power_summary
    stats = logger.stats_summary

    # Get average packet size in flits and convert to bytes
    avg_packet_length_flits = stats.accepted_packet_length_avg
    flit_size_bytes = ps.flit_width_bits / 8
    avg_packet_size_bytes = avg_packet_length_flits * flit_size_bytes

    # Calculate energy for each packet based on its latency
    packet_energies = {
        'WRITE': [],
        'WRITE_REQ': [],
        'REPLY': [],
        'READ': [],
        'READ_REQ': [],
        'ALL': []
    }

    for packet in packet_latencies:
        comm_type = packet['communication_type']
        latency_cycles = packet['latency']

        # Energy consumed by this packet (assumes constant power during transmission)
        packet_energy_J = ps.total_power * latency_cycles / ps.fclk

        # Energy per byte for this packet (using average packet size)
        energy_per_byte_J = packet_energy_J / avg_packet_size_bytes if avg_packet_size_bytes > 0 else 0

        # Store by communication type
        packet_energies['ALL'].append({
            'packet_id': packet['packet_id'],
            'type': comm_type,
            'latency': latency_cycles,
            'energy_J': packet_energy_J,
            'energy_per_byte_J': energy_per_byte_J
        })

        if comm_type in packet_energies:
            packet_energies[comm_type].append(energy_per_byte_J)

    # Calculate averages for each communication type
    result = {
        'avg_packet_size_bytes': avg_packet_size_bytes,
        'avg_packet_length_flits': avg_packet_length_flits,
        'flit_size_bytes': flit_size_bytes,
        'total_packets_analyzed': len(packet_latencies)
    }

    for comm_type in ['WRITE', 'WRITE_REQ', 'REPLY', 'READ', 'READ_REQ']:
        if packet_energies[comm_type]:
            energies = packet_energies[comm_type]
            result[f'{comm_type}_energy_per_byte_J'] = np.mean(energies)
            result[f'{comm_type}_energy_per_byte_pJ'] = np.mean(energies) * 1e12
            result[f'{comm_type}_energy_per_byte_std_pJ'] = np.std(energies) * 1e12
            result[f'{comm_type}_packet_count'] = len(energies)
        else:
            result[f'{comm_type}_energy_per_byte_J'] = 0
            result[f'{comm_type}_energy_per_byte_pJ'] = 0
            result[f'{comm_type}_energy_per_byte_std_pJ'] = 0
            result[f'{comm_type}_packet_count'] = 0

    # Also calculate overall average
    if packet_energies['ALL']:
        all_energies = [p['energy_per_byte_J'] for p in packet_energies['ALL']]
        result['overall_energy_per_byte_J'] = np.mean(all_energies)
        result['overall_energy_per_byte_pJ'] = np.mean(all_energies) * 1e12
        result['overall_energy_per_byte_std_pJ'] = np.std(all_energies) * 1e12

    result['packet_energies'] = packet_energies

    return result


def calculate_noc_energy_per_byte(logger, num_nodes=None, config_data=None):
    """
    Calculate actual NoC energy per byte from Booksim2 power and stats summaries

    Parameters:
    -----------
    logger : EventLogger
        Logger object with power_summary and stats_summary
    num_nodes : int, optional
        Number of nodes in the NoC. If not provided, will try to extract from config_data
    config_data : dict, optional
        Configuration data containing 'k' parameter (num_nodes = k * k)

    Returns:
    --------
    dict with energy metrics and breakdown
    """
    if not logger.has_power_summary:
        raise ValueError("Logger does not have power summary. Set sim_power=1 in config.")

    if not logger.has_stats_summary:
        raise ValueError("Logger does not have stats summary. Set logger=1 in config.")

    # Determine number of nodes
    if num_nodes is None:
        if config_data is not None:
            # Try to extract k from config (could be at root or in 'arch' section)
            k = None
            if 'k' in config_data:
                k = config_data['k']
            elif 'arch' in config_data and 'k' in config_data['arch']:
                k = config_data['arch']['k']

            if k is not None:
                num_nodes = k * k
            else:
                # Default to 64 (8x8 grid)
                num_nodes = 64
                print(f"Warning: num_nodes not provided and 'k' not found in config. Using default: {num_nodes}")
        else:
            num_nodes = 64
            print(f"Warning: num_nodes and config_data not provided. Using default: {num_nodes}")

    ps = logger.power_summary
    stats = logger.stats_summary

    # Calculate total NoC energy (in Joules)
    total_noc_energy_J = ps.total_power * ps.completion_time_cycles / ps.fclk

    # Calculate total bytes transmitted
    # Total packets = accepted_packet_rate * time * num_nodes
    total_packets = stats.accepted_packet_rate_avg * stats.time_elapsed_cycles * num_nodes

    # Average packet length in flits
    avg_packet_length_flits = stats.accepted_packet_length_avg

    # Flit size in bytes
    flit_size_bytes = ps.flit_width_bits / 8

    # Total bytes transmitted
    total_bytes = total_packets * avg_packet_length_flits * flit_size_bytes

    # Energy per byte (in Joules)
    energy_per_byte_J = total_noc_energy_J / total_bytes if total_bytes > 0 else 0

    # Breakdown by power component
    channel_power = (ps.channel_wire_power + ps.channel_clock_power +
                     ps.channel_retiming_power + ps.channel_leakage_power)
    router_power = (ps.input_read_power + ps.input_write_power +
                    ps.input_leakage_power + ps.switch_power +
                    ps.switch_control_power + ps.switch_leakage_power +
                    ps.output_dff_power + ps.output_clk_power +
                    ps.output_control_power)

    channel_energy_per_byte_J = (channel_power * ps.completion_time_cycles / ps.fclk / total_bytes
                                  if total_bytes > 0 else 0)
    router_energy_per_byte_J = (router_power * ps.completion_time_cycles / ps.fclk / total_bytes
                                 if total_bytes > 0 else 0)

    result = {
        # Total energy
        'total_noc_energy_J': total_noc_energy_J,
        'total_noc_energy_uJ': total_noc_energy_J * 1e6,
        'total_noc_energy_nJ': total_noc_energy_J * 1e9,

        # Traffic stats
        'total_packets': total_packets,
        'avg_packet_length_flits': avg_packet_length_flits,
        'flit_size_bytes': flit_size_bytes,
        'total_bytes': total_bytes,

        # Energy per byte
        'energy_per_byte_J': energy_per_byte_J,
        'energy_per_byte_pJ': energy_per_byte_J * 1e12,

        # Breakdown
        'channel_energy_per_byte_J': channel_energy_per_byte_J,
        'channel_energy_per_byte_pJ': channel_energy_per_byte_J * 1e12,
        'router_energy_per_byte_J': router_energy_per_byte_J,
        'router_energy_per_byte_pJ': router_energy_per_byte_J * 1e12,
    }

    return result


def get_updated_energy_params(logger, packet_latencies=None, num_nodes=None, config_data=None, use_per_packet=True):
    """
    Get updated energy parameters using actual NoC simulation data

    Parameters:
    -----------
    logger : EventLogger
        Logger with power and stats summaries
    packet_latencies : List[Dict], optional
        List of packet latency dictionaries. Required if use_per_packet=True
    num_nodes : int, optional
        Number of nodes in the NoC. Used if use_per_packet=False
    config_data : dict, optional
        Configuration data containing 'k' parameter
    use_per_packet : bool, optional
        If True, use per-packet energy calculation (more accurate). Default: True

    Returns:
    --------
    tuple: (updated_params dict, noc_energy dict)
    """
    if use_per_packet and packet_latencies:
        # Use per-packet energy calculation (more accurate)
        noc_energy = calculate_noc_energy_per_byte_per_packet(logger, packet_latencies, config_data)

        # Extract energy per byte for each communication type
        write_energy = noc_energy.get('WRITE_energy_per_byte_J', noc_energy.get('overall_energy_per_byte_J', 5e-12))
        write_req_energy = noc_energy.get('WRITE_REQ_energy_per_byte_J', write_energy)
        reply_energy = noc_energy.get('REPLY_energy_per_byte_J', write_energy)

        updated_params = {
            "COMP_OP":    {"energy_per_flop": 1.1e-12},   # From Horowitz
            "WRITE":      {"energy_per_byte": write_energy},  # From per-packet NoC simulation
            "WRITE_REQ":  {"energy_per_byte": write_req_energy},  # From per-packet NoC simulation
            "REPLY":      {"energy_per_byte": reply_energy},  # From per-packet NoC simulation
            "SRAM_READ":  {"energy_per_byte": 10.5e-12},  # From Horowitz
            "SRAM_WRITE": {"energy_per_byte": 12.5e-12},  # From Horowitz
            "DEFAULT":    {"energy_per_byte": 1.1e-12, "energy_per_flop": 0.4e-12}
        }
    else:
        # Fall back to aggregate calculation
        noc_energy = calculate_noc_energy_per_byte(logger, num_nodes, config_data)

        # Use calculated energy per byte for all network communication
        energy_per_byte = noc_energy['energy_per_byte_J']

        updated_params = {
            "COMP_OP":   {"energy_per_flop": 1.1e-12},   # From Horowitz
            "WRITE":     {"energy_per_byte": energy_per_byte},  # From NoC simulation
            "WRITE_REQ": {"energy_per_byte": energy_per_byte},  # From NoC simulation
            "REPLY":     {"energy_per_byte": energy_per_byte},  # From NoC simulation
            "SRAM_READ":  {"energy_per_byte": 10.5e-12},  # From Horowitz
            "SRAM_WRITE": {"energy_per_byte": 12.5e-12},  # From Horowitz
            "DEFAULT":    {"energy_per_byte": 1.1e-12, "energy_per_flop": 0.4e-12}
        }

    return updated_params, noc_energy


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


def export_latency_energy_details_to_csv(
    parallel_analysis: Dict,
    noc_energy_result: Dict,
    logger,
    computation_times: List[Dict] = None,
    packet_latencies: List[Dict] = None,
    energy_params: Dict = None,
    output_path: str = "latency_energy_details.csv",
    config_data: Dict = None
) -> pd.DataFrame:
    """
    Export comprehensive latency and energy details to CSV file.

    Separates NoC energy (from Booksim2 power model) and PE energy (from computation events).

    Parameters:
    -----------
    parallel_analysis : Dict
        Result from analyze_parallel_execution_time()
    noc_energy_result : Dict
        Result from calculate_noc_energy_per_byte_per_packet()
    logger : EventLogger
        Logger object with power summary
    computation_times : List[Dict], optional
        List of computation time dictionaries from analyze_computation_times()
    packet_latencies : List[Dict], optional
        List of packet latency dictionaries from analyze_packet_latencies()
    energy_params : Dict, optional
        Energy parameters dict (if None, uses DEFAULT_ENERGY_PARAMS)
    output_path : str
        Path to output CSV file
    config_data : Dict, optional
        Configuration data for additional context

    Returns:
    --------
    pd.DataFrame with the exported data
    """
    if not logger.has_power_summary:
        raise ValueError("Logger does not have power summary")

    ps = logger.power_summary

    if energy_params is None:
        energy_params = DEFAULT_ENERGY_PARAMS.copy()

    # ==========================================
    # NoC Energy (from Booksim2 power model)
    # ==========================================
    # Total NoC energy from power summary - this is ONLY for routers/channels
    noc_total_energy_J = ps.total_power * ps.completion_time_cycles / ps.fclk

    # ==========================================
    # PE Energy (from computation events)
    # ==========================================
    pe_energy_J = 0.0

    if computation_times:
        # Estimate PE energy from computation operations
        comp_flop_energy = energy_params.get('COMP_OP', {}).get('energy_per_flop', 1.1e-12)

        for comp in computation_times:
            # Use duration as flop estimate (or computation_time if available)
            estimated_flops = comp.get('computation_time', comp['duration'])
            pe_energy_J += estimated_flops * comp_flop_energy

    # ==========================================
    # Data Flow Energy (from packet events)
    # ==========================================
    dataflow_energy_J = 0.0

    if packet_latencies and noc_energy_result and 'packet_energies' in noc_energy_result:
        # Sum up actual energy consumed by all packets
        all_packets = noc_energy_result['packet_energies'].get('ALL', [])
        for packet_info in all_packets:
            dataflow_energy_J += packet_info['energy_J']
    else:
        # Fallback: use NoC total energy as data flow energy
        dataflow_energy_J = noc_total_energy_J

    # Total energy = PE energy + Data flow energy
    total_energy_J = pe_energy_J + dataflow_energy_J

    # Create summary row
    data = {
        # Latency metrics (cycles)
        'overall_latency_cycles': parallel_analysis['total_simulation_cycles'],
        'comp_cycles': parallel_analysis['computation_occupied_cycles'],
        'data_flow_cycles': parallel_analysis['packet_occupied_cycles'],
        'overlapping_cycles': parallel_analysis['overlap_cycles'],
        'idle_cycles': parallel_analysis['idle_cycles'],

        # Energy metrics (Joules)
        'energy_PEs_J': pe_energy_J,
        'energy_data_flow_J': dataflow_energy_J,
        'total_energy_J': total_energy_J,

        # Energy metrics (microJoules)
        'energy_PEs_uJ': pe_energy_J * 1e6,
        'energy_data_flow_uJ': dataflow_energy_J * 1e6,
        'total_energy_uJ': total_energy_J * 1e6,

        # Percentage breakdowns
        'comp_percentage': parallel_analysis['computation_percentage'],
        'data_flow_percentage': parallel_analysis['packet_percentage'],
        'overlap_percentage': parallel_analysis['overlap_percentage'],
        'idle_percentage': parallel_analysis['idle_percentage'],

        # NoC energy details (if available)
        'noc_energy_per_byte_pJ': noc_energy_result.get('overall_energy_per_byte_pJ', 0) if noc_energy_result else 0,
        'total_packets_analyzed': noc_energy_result.get('total_packets_analyzed', 0) if noc_energy_result else 0,
        'avg_packet_size_bytes': noc_energy_result.get('avg_packet_size_bytes', 0) if noc_energy_result else 0,

        # Power details
        'noc_power_W': ps.total_power,
        'clock_freq_Hz': ps.fclk,

        # Parallelism metrics
        'max_parallel_computations': parallel_analysis['max_parallel_computations'],
        'avg_parallel_computations': parallel_analysis['avg_parallel_computations'],
        'parallelism_speedup': parallel_analysis['parallelism_speedup'],

        # Computation details
        'total_computations': len(computation_times) if computation_times else 0,
        'total_packets': len(packet_latencies) if packet_latencies else 0,
    }

    # Add per-packet-type energy if available
    if noc_energy_result:
        for comm_type in ['WRITE', 'WRITE_REQ', 'REPLY', 'READ', 'READ_REQ']:
            data[f'{comm_type}_energy_per_byte_pJ'] = noc_energy_result.get(f'{comm_type}_energy_per_byte_pJ', 0)
            data[f'{comm_type}_packet_count'] = noc_energy_result.get(f'{comm_type}_packet_count', 0)

    # Create DataFrame
    df = pd.DataFrame([data])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Exported latency and energy details to: {output_path}")

    return df


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

