import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_metric(result_files, exp_names, metric_name, output_file='plot.png'):
    # Set default font sizes
    plt.rcParams.update({
        'font.size': 36,  # Default font size
        'axes.titlesize': 36,  # Title font size
        'axes.labelsize': 36,  # Axis label font size
        'xtick.labelsize': 30,  # X-axis tick label font size
        'ytick.labelsize': 30,  # Y-axis tick label font size
        'legend.fontsize': 30,  # Legend font size
    })
    
    plt.figure(figsize=(10, 6))
    
    for result_file, exp_name in zip(result_files, exp_names):
        # Load the metrics data
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Extract n values and corresponding scores
        n_values = []
        scores = []
        for key in data.keys():
            if key.startswith(f'{metric_name}@'):
                n = int(key.split('@')[1])
                n_values.append(n)
                scores.append(data[key])

        # Sort the values by n
        n_values, scores = zip(*sorted(zip(n_values, scores)))

        # Plot the data
        plt.plot(n_values, scores, 'o-', linewidth=3, markersize=12, label=exp_name)

    # Set log scale for x-axis with base 2
    plt.xscale('log', base=2)
    plt.xticks(n_values, n_values)

    # Add labels and title
    plt.xlabel('n')
    plt.ylabel('Score')

    # Add grid and legend
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot metrics from JSON files')
    parser.add_argument('--result_files', nargs='+', required=True, help='List of result JSON files')
    parser.add_argument('--exp_names', nargs='+', required=True, help='List of experiment names for legend')
    parser.add_argument('--metric_name', default='best_score_label', help='Metric name to plot (default: best_score_label)')
    parser.add_argument('--output_file', default='plot.png', help='Output file name (default: plot.png)')
    
    args = parser.parse_args()
    
    if len(args.result_files) != len(args.exp_names):
        raise ValueError("Number of result files must match number of experiment names")
    
    plot_metric(args.result_files, args.exp_names, args.metric_name, args.output_file) 
