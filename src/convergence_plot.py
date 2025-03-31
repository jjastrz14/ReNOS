import matplotlib.pyplot as plt
import numpy as np
from dirs import MAIN_DIR

def plot_convergence():
    prefix = "/data"
    folder = "/ACO"
    folder = prefix + folder
    stats = np.load(MAIN_DIR + folder + "/statistics.npy", allow_pickle=True).item()
    mean_values = np.array(stats["mdn"])
    std_values = np.array(stats["std"])
    best_values = np.array([stats["best"][i][2] for i in range(len(stats["best"]))])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # add mean line
    ax.plot(mean_values, label="iteration mean", color = "lightseagreen")
    # fill between the -std and +std
    ax.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2, color = "mediumturquoise")
    # add best line
    ax.plot(best_values, label="iteration best", color = "lightcoral")
    ax.set_xlim(0, len(mean_values))
    ax.tick_params(direction='in')
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.legend(fontsize=14)
    plt.xlabel("Iterations", fontdict={"size": 14})
    plt.ylabel("Latency estimation", fontdict={"size": 14})
    plt.title("ACO convergence", fontdict={"size": 16})
    #plt.show()

    fig.savefig("visual/convergence.png", dpi=300)

def plot_convergence_step_functions():
    # Plot a bunch of convergence plots in the same figure
    prefix = "/data"
    # folders = ["/ACO_reduced", "/ACO_heuristic", "/GA_reduced"]
    # labels = [r"ACO$_{random}$", r"ACO$_{improved}$", r"GA"]
    folders = ["/ACO_1", "/ACO_01", "/ACO_001", "/ACO_0001"]
    labels = [r"ACO_10x", r"ACO_100x", r"ACO_1000x" , r"ACO_10000x"]
    #folders = ["/GA_1", "/GA_001", "/GA_0001"]
    #labels = [r"GA_10x", r"ACO_1000x" , r"ACO_10000x"]
    stats = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ["lightcoral", "lightseagreen", "mediumpurple", "mediumturquoise"]
    for i,folder in enumerate(folders):
        folder = prefix + folder
        stats.append(np.load(MAIN_DIR + folder + "/statistics.npy", allow_pickle=True).item())
        mean_values = np.array(stats[-1]["mdn"])
        std_values = np.array(stats[-1]["std"])
        best_values = np.array(stats[-1]["best"])
        best_values = np.array(stats[-1]["absolute_best"])
        # best_values = np.array([stats["absolute_best"][i][2] for i in range(1,len(stats["best"]))])

        if "GA" in folder:
            best_values = np.array(stats[-1]["best"])

        name = folder.split("/")[-1]
        

        # add best line
        ax.plot(best_values, label=labels[i], color = colors[i], linewidth=3)
    #ax.set_xlim(0, len(mean_values))
    ax.tick_params(direction='in')
    #ax.spines['top'].set_linewidth(2)
    #ax.spines['right'].set_linewidth(2)
    #ax.spines['bottom'].set_linewidth(2)
    #ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.tick_params(axis='both', which='major', labelsize=14)


    plt.legend(fontsize=14)
    plt.xlabel("Iterations", fontdict={"size": 16})
    plt.ylabel("Latency [cycles]", fontdict={"size": 16})
    # plt.title("convergence", fontdict={"size": 16})
    fig.savefig("visual/convergence_ACO_comp.png", dpi=300)
    
if __name__ == "__main__":
    #plot_convergence()
    plot_convergence_step_functions()