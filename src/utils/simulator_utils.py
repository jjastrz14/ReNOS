import simulator_stub as ss
import fast_analytical_simulator_stub as ssam
from visualizer import plot_timeline
"""
Utility functions for simulator comparison and analysis
"""

def compare_simulators_and_visualize(best_solution_path: str, output_dir: str, algorithm_name: str, 
                                    timestamp: str, verbose: bool = True):
    """
    Compare analytical model vs BookSim2 simulator and create visualizations
    
    Args:
        best_solution_path: Path to the best solution JSON file
        output_dir: Directory to save outputs
        algorithm_name: Name of the algorithm (ACO, GA, etc.)
        timestamp: Timestamp string for file naming
        verbose: Whether to create timeline visualization
    
    Returns:
        dict: Results containing latencies and comparison metrics
    """
    
    results = {}
    timeline_path = f"{output_dir}/timeline_{algorithm_name}_{timestamp}.png"
    analytical_timeline_path = f"{output_dir}/analytical_timeline_{algorithm_name}_{timestamp}.png"
    utilization_path = f"{output_dir}/utilization_{algorithm_name}_{timestamp}.png"

    if verbose:
        print("Visualizing the best path...\n")
        # Visualize the best path
        plot_timeline(best_solution_path, timeline_path=timeline_path, verbose=False)
    else: 
        #run just NoC complex simulator and get the latency
        print("Running the NoC simulator on the best path found...\n")
    
    # Run analytical simulator
    #print("Running analytical model...")
    #stub_anal = ssam.FastAnalyticalSimulatorStub()
    #total_latency, logger_anal = stub_anal.run_simulation(best_solution_path, dwrap=False)
    #print(f"Analytical model result: {total_latency} cycles")
    #results['analytical_latency'] = total_latency
    
    # Run BookSim2 simulator for comparison
    print("Running BookSim2 simulator for comparison...")
    stub = ss.SimulatorStub()
    booksim_result, logger = stub.run_simulation(best_solution_path, dwrap=False)
    print(f"BookSim2 result: {booksim_result} cycles")
    results['booksim_latency'] = booksim_result
    
    return results