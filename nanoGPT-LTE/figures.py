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

if __name__ == "__main__":
    datafiles = [
        datafile(False, "", -1, -1, "2024-04-30-23-29-30"),
        datafile(True, "mhlora", 1, 4, "2024-04-30-18-43-36"),
        datafile(True, "mhlora", 1, 64, "2024-04-30-18-53-05"),
        datafile(True, "mhlora", 4, 4, "2024-04-30-19-03-23"),
        datafile(True, "mhlora", 4, 64, "2024-04-30-19-14-24"),
        datafile(True, "dmp", 1, 4, "2024-04-30-23-38-45"),
        datafile(True, "dmp", 1, 64, "2024-04-30-23-47-58"),
        datafile(True, "dmp", 4, 4, "2024-04-30-23-57-14"),
        datafile(True, "dmp", 4, 64, "2024-05-01-00-06-37"), 
    ]

    ys = [None] * len(datafiles)

    for i, datafile in enumerate(datafiles):
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

        ys[i] = pd.read_csv(file_path)['val_loss']
        assert len(ys[i]) == 21
    
    # Create some data for the line plots
    x = range(0, 201, 10)
    
    # Plot the line plots
    for i, y in enumerate(ys):
        if datafiles[i].has_lte == False:
            label = "Without LoRA"
        else:
            label = f"{'Parallel-Merge' if datafiles[i].lte_mode == 'dmp' else 'Sequential'}" + \
                f" (H {datafiles[i].lte_heads} R {datafiles[i].lora_rank})"
        
        plt.plot(x, y, label=label)

        # Set y axis limits
        plt.ylim(6, 10)

        # Add axis labels
        plt.xlabel("Iteration")
        plt.ylabel("Test Loss")

        # Add a legend
        plt.legend()
        
    # Show the plot
    plt.savefig("figures/val_loss.png")