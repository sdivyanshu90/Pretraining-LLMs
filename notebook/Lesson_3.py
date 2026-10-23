# %% [markdown]
# Source notebook: Lesson_3.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 3: Data Packaging
# ## 1. Tokenizing and creating input_ids
# 
# Start by loading the dataset from the previous lesson:

# %%
# Notebook cell 2
import datasets

dataset = datasets.load_dataset(
    "parquet", 
    data_files="./data/preprocessed_dataset.parquet", 
    split="train"
)
print(dataset)

# %% [markdown]
# Use the `shard` method of the Hugging Face `Dataset` object to split the dataset into 10 smaller pieces, or *shards* (think shards of broken glass). You can read more about sharding at [this link](https://huggingface.co/docs/datasets/en/process#shard).

# %%
# Notebook cell 4
dataset = dataset.shard(num_shards=10, index=0)
print(dataset)

# %% [markdown]
# Load the tokenizer and try it out:

# %%
# Notebook cell 6
from transformers import AutoTokenizer
model_path_or_name = "./models/SOLAR-10.7B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name, 
    use_fast=False
)

# %%
# Notebook cell 7
tokenizer.tokenize("I'm a short sentence")

# %% [markdown]
# Create a helper function:

# %%
# Notebook cell 9
def tokenization(example):
    # Tokenize
    tokens = tokenizer.tokenize(example["text"])

    # Convert tokens to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add <bos>, <eos> tokens to the front and back of tokens_ids 
    # bos: begin of sequence, eos: end of sequence
    token_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]
    example["input_ids"] = token_ids

    # We will be using this column to count the total number of tokens 
    # in the final dataset
    example["num_tokens"] = len(token_ids)
    return example

# %% [markdown]
# Tokenize all the examples in the pretraining dataset:

# %%
# Notebook cell 11
dataset = dataset.map(tokenization, load_from_cache_file=False)
print(dataset)

# %%
# Notebook cell 12
sample = dataset[3]

print("text", sample["text"][:30]) # 
print("\ninput_ids", sample["input_ids"][:30])
print("\nnum_tokens", sample["num_tokens"])

# %% [markdown]
# Check the total number of tokens in the dataset:

# %%
# Notebook cell 14
import numpy as np
np.sum(dataset["num_tokens"])

# %% [markdown]
# ## 2. Packing the data

# %% [markdown]
# ![Packing data for training](./data_packing.png)

# %% [markdown]
# Concatenate input_ids for all examples into a single list:

# %%
# Notebook cell 18
input_ids = np.concatenate(dataset["input_ids"])
print(len(input_ids))

# %%
# Notebook cell 19
max_seq_length = 32

# %%
# Notebook cell 20
total_length = len(input_ids) - len(input_ids) % max_seq_length
print(total_length)

# %% [markdown]
# Discard extra tokens from end of the list so number of tokens is exactly divisible by `max_seq_length`:

# %%
# Notebook cell 22
input_ids = input_ids[:total_length]
print(input_ids.shape)

# %%
# Notebook cell 23
input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)
input_ids_reshaped.shape

# %%
# Notebook cell 24
type(input_ids_reshaped)

# %% [markdown]
# Convert to Hugging Face dataset:

# %%
# Notebook cell 26
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)
print(packaged_pretrain_dataset)

# %% [markdown]
# ## 3. Save the packed dataset to disk

# %%
# Notebook cell 28
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")

# %%
# Notebook cell 29


# %%
# Notebook cell 30


# %%
# Notebook cell 31


# %%
# Notebook cell 32


# %%
# Notebook cell 33


# %%
# Notebook cell 34


# %%
# Notebook cell 35


# %%
# Notebook cell 36


# %%
# Notebook cell 37


# %%
# Notebook cell 38


# %%
# Notebook cell 39


# %%
# Notebook cell 40


# %%
# Notebook cell 41

