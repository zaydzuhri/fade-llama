import os
import numpy as np
from datasets import load_dataset
import csv
import sys
sys.path.append(os.path.abspath(os.path.join('../../', 'llama')))
from tokenizer import Tokenizer

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# join all text
print("joining all text...")
data = '\n'.join(dataset['text'])

print("sample: ", data[:1000])

n = len(data)
# encode with llama tokenizer
print("encoding with llama tokenizer...")
tokenizer = Tokenizer(model_path='../../tokenizer.model')
test_ids = tokenizer.encode(data, eos=False, bos=False)
print(f"test has {len(test_ids):,} tokens")

# export to bin files
print("exporting to bin files...")
test_ids = np.array(test_ids, dtype=np.uint16)
test_ids.tofile('test.bin')
