import logging
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

@dataclass
class datafile:
    has_lte: bool
    lte_mode: str
    lte_heads: int
    lora_rank: int
    file_name: str

config2_datafiles = {
    "wolora": datafile(False, "", -1, -1, "2024-04-30-23-29-30"),
    "sh1r4": datafile(True, "mhlora", 1, 4, "2024-04-30-18-43-36"),
    "sh1r64": datafile(True, "mhlora", 1, 64, "2024-04-30-18-53-05"),
    "sh4r4": datafile(True, "mhlora", 4, 4, "2024-04-30-19-03-23"),
    "sh4r64": datafile(True, "mhlora", 4, 64, "2024-04-30-19-14-24"),
    "ph1r4": datafile(True, "dmp", 1, 4, "2024-04-30-23-38-45"),
    "ph1r64": datafile(True, "dmp", 1, 64, "2024-04-30-23-47-58"),
    "ph4r4": datafile(True, "dmp", 4, 4, "2024-04-30-23-57-14"),
    "ph4r64": datafile(True, "dmp", 4, 64, "2024-05-01-00-06-37"), 
}

config2_loss_mem_datafiles = [ 
    config2_datafiles["wolora"],
    config2_datafiles["sh1r4"],
    config2_datafiles["sh1r64"],
    config2_datafiles["sh4r4"],
    config2_datafiles["sh4r64"],
    config2_datafiles["ph1r4"],
    config2_datafiles["ph1r64"],
    config2_datafiles["ph4r4"],
    config2_datafiles["ph4r64"],
]

def get_measures():
    config_dfs = [None] * len(config2_loss_mem_datafiles)
    data_dfs = [None] * len(config2_loss_mem_datafiles)

    for i, datafile in enumerate(config2_loss_mem_datafiles):
        logging.info(f"Processing {datafile.file_name}")

        file_path = f"log/{datafile.file_name[:10]}/{datafile.file_name}.csv"
        config_file_path = file_path[:-4] + "_config.csv"
        config_df = pd.read_csv(config_file_path)
        if config_df.iloc[0]['wrap_lte'] == False:
            assert datafile.has_lte == False
        else:
            assert datafile.has_lte == True
            assert config_df.iloc[0]['lte_mode'] == datafile.lte_mode
            assert config_df.iloc[0]['lte_heads'] == datafile.lte_heads
            assert config_df.iloc[0]['lora_r'] == datafile.lora_rank

        config_dfs[i] = config_df
        data_dfs[i] = pd.read_csv(file_path)
        assert len(data_dfs[i]) == 21

    return config_dfs, data_dfs

def plot_val_loss():
    ys = [None] * len(config2_loss_mem_datafiles)
    
    _, data_dfs = get_measures()
    for i in range(len(ys)):
        ys[i] = data_dfs[i]['val_loss']

    # Create some data for the line plots
    x = range(0, 201, 10)
    
    # Plot the line plots
    handles = []
    labels = []
    for i, y in enumerate(ys):
        if config2_loss_mem_datafiles[i].has_lte == False:
            label = "W/o LoRA"
        else:
            label = f"{'Par-Merge' if config2_loss_mem_datafiles[i].lte_mode == 'dmp' else 'Seq'}" + \
                f" (H{config2_loss_mem_datafiles[i].lte_heads} R{config2_loss_mem_datafiles[i].lora_rank})"
        
        line, = plt.plot(x, y, label=label, linewidth=2.5)
        handles.append(line)
        labels.append(label)

    order = [3, 4, 6, 5, 2, 8, 1, 0, 7] # from highest to lowest loss at end of training
    ordered_handles = [handles[i] for i in order]
    ordered_labels = [labels[i] for i in order]

    #Set font size
    plt.rcParams.update({'font.size': 16})

    # Set y axis limits
    plt.ylim(6.5, 10)

    # Add axis labels
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Test Loss", fontsize=16)

    # Set font size of ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # show legend on the right of the plot 
    # plt.legend(ordered_handles, ordered_labels, loc='center right', 
    #             bbox_to_anchor=(1.6, 0.5), fontsize=12)
    plt.legend(ordered_handles, ordered_labels, loc='upper right', 
                fontsize=12)

    # Show the plot
    plt.savefig("figures/val_loss.png", bbox_inches='tight', dpi=300)
    plt.savefig("figures/val_loss.pdf", bbox_inches='tight', dpi=300)


def plot_mem_time():
    config_dfs, data_dfs = get_measures()
    params_trained = [124.083, 95.955, 99.273, 96.619, 109.89, 95.955, 99.273, 96.619, 109.89]
    labels = []
    iter_times = []

    for ddf in data_dfs:
        # cdf does not have train_params :'(. collecting from wandb.
        # params_trained[i] = cdf.iloc[0]['params_trained']
        iter_times.append(ddf['train_time'].tolist()[-1] / 200)
    
    for datafile in config2_loss_mem_datafiles:
        if datafile.has_lte == False:
            labels.append("W/o LoRA")
        else:
            labels.append(f"{'Par-Merge' if datafile.lte_mode == 'dmp' else 'Seq'}" + \
                f" (H{datafile.lte_heads} R{datafile.lora_rank})")
    
    #Set font size
    plt.rcParams.update({'font.size': 14})

    # Create scatter plot
    plt.scatter(iter_times, params_trained)

    # Add labels to data points
    label_fontsize=12
    plt.annotate(labels[0], (iter_times[0], params_trained[0]), fontsize=label_fontsize, 
        ha='center', va='bottom')
    plt.annotate(labels[1], (iter_times[1], params_trained[1]), fontsize=label_fontsize, 
        xytext=(5, -3), textcoords='offset points')
    plt.annotate(labels[2], (iter_times[2], params_trained[2]), fontsize=label_fontsize, 
        xytext=(5, -3), textcoords='offset points')
    plt.annotate(labels[3], (iter_times[3], params_trained[3]), fontsize=label_fontsize, 
        ha='center', va='bottom')
    plt.annotate(labels[4], (iter_times[4], params_trained[4]), fontsize=label_fontsize, 
        ha='center', va='bottom')
    plt.annotate(labels[5], (iter_times[5], params_trained[5]), fontsize=label_fontsize, 
        xytext=(-40, -15), textcoords='offset points')
    plt.annotate(labels[6], (iter_times[6], params_trained[6]), fontsize=label_fontsize, 
        ha='center', va='bottom', xytext=(0.01, 3), textcoords='offset points')
    plt.annotate(labels[7], (iter_times[7], params_trained[7]), fontsize=label_fontsize, 
        ha='center', va='bottom', xytext=(0, 1.2), textcoords='offset points')
    plt.annotate(labels[8], (iter_times[8], params_trained[8]), fontsize=label_fontsize, 
        ha='center', va='bottom')

    # Add axis labels
    plt.xlabel("Avg. Iteration Time (s)")
    plt.ylabel("Parameters Trained (M)")

    # Set axis limit
    plt.xlim(1.15, 1.6)
    plt.ylim(90, 130)

    # Add light grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Show only left and bottom axis
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig("figures/mem_time.png", dpi=300)

if __name__ == "__main__":
    plot_mem_time()