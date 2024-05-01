import os
import requests

import tiktoken
import numpy as np
import pandas as pd

files = ['test-00000-of-00001.parquet', 'train-00000-of-00003.parquet', 
               'train-00001-of-00003.parquet', 'train-00002-of-00003.parquet', 
               'validation-00000-of-00001.parquet']

def download_dataset():
    for fn in files:
        input_file_path = os.path.join(os.path.dirname(__file__), fn)
        if not os.path.exists(input_file_path):
            data_url = f'https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/{fn}'
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text) # Does this work for parquet?

def create_bin():
    train_data = ""
    val_data = ""
    test_data = ""
    for fn in files:
        data = pd.read_parquet(fn)  # columns: article, highlights, id
        if 'train' in fn:
            train_data += ''.join(data['article'])
        elif 'validation' in fn:
            val_data += ''.join(data['article'])
        elif 'test' in fn:
            test_data += ''.join(data['article'])

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    test_ids = enc.encode_ordinary(test_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# train.bin has 249182164 tokens
# val.bin has 11302606 tokens
# test.bin has 9841240 tokens

if __name__ == '__main__':
    download_dataset()
    create_bin()
