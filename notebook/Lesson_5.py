# %% [markdown]
# Source notebook: Lesson_5.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 5. Model training
# 
# Pretraining is very expensive! Please check costs carefully before starting a pretraining project.
# 
# You can get a rough estimate your training job cost using [this calculator](https://huggingface.co/training-cluster) from Hugging Face. For training on other infrastructure, e.g. AWS or Google Cloud, please consult those providers for up to date cost estimates. 

# %%
# Notebook cell 2
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Load the model to be trained
# 
# Load the upscaled model from the previous lesson:

# %%
# Notebook cell 4
import torch
from transformers import AutoModelForCausalLM

pretrained_model = AutoModelForCausalLM.from_pretrained(
    "./models/TinySolar-308m-4k-init",
    device_map="cpu", 
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# %%
# Notebook cell 5
pretrained_model

# %% [markdown]
# ## 2. Load dataset
# 
# Here you'll update two methods on the `Dataset` object to allow it to interface with the trainer. These will be applied when you specify the dataset you created in Lesson 3 as the training data in the next section.
# 
# Note that the code has additional comment strings that don't appear in the video. These are to help you understand what each part of the code is doing.

# %%
# Notebook cell 7
import datasets
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, split="train"):
        """Initializes the custom dataset object."""
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset 
        at the specified index
        """
        # Convert the lists to a LongTensor for PyTorch
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])

        # Return the sample as a dictionary
        return {"input_ids": input_ids, "labels": labels}

# %% [markdown]
# ## 3. Configure Training Arguments
# 
# Here you set up the training run. The training dataset you created in Lesson 3 is specified in the Dataset configuration section.
# 
# Note: there are comment strings in the cell below that don't appear in the video. These have been included to help you understand what each parameter does.

# %%
# Notebook cell 9
from dataclasses import dataclass, field
import transformers

@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(                           # Dataset configuration
        default="./parquet/packaged_pretrain_dataset.parquet")
    num_proc: int = field(default=1)                     # Number of subprocesses for data preprocessing
    max_seq_length: int = field(default=32)              # Maximum sequence length

    # Core training configurations
    seed: int = field(default=0)                         # Random seed for initialization, ensuring reproducibility
    optim: str = field(default="adamw_torch")            # Optimizer, here it's AdamW implemented in PyTorch
    max_steps: int = field(default=300)                   # Number of maximum training steps
    per_device_train_batch_size: int = field(default=2)  # Batch size per device during training

    # Other training configurations
    learning_rate: float = field(default=5e-5)           # Initial learning rate for the optimizer
    weight_decay: float = field(default=0)               # Weight decay
    warmup_steps: int = field(default=10)                # Number of steps for the learning rate warmup phase
    lr_scheduler_type: str = field(default="linear")     # Type of learning rate scheduler
    gradient_checkpointing: bool = field(default=True)   # Enable gradient checkpointing to save memory
    dataloader_num_workers: int = field(default=2)       # Number of subprocesses for data loading
    bf16: bool = field(default=True)                     # Use bfloat16 precision for training on supported hardware
    gradient_accumulation_steps: int = field(default=1)  # Number of steps to accumulate gradients before updating model weights

    # Logging configuration
    logging_steps: int = field(default=3)                # Frequency of logging training information
    report_to: str = field(default="none")               # Destination for logging (e.g., WandB, TensorBoard)

    # Saving configuration
    # save_strategy: str = field(default="steps")          # Can be replaced with "epoch"
    # save_steps: int = field(default=3)                   # Frequency of saving training checkpoint
    # save_total_limit: int = field(default=2)             # The total number of checkpoints to be saved

# %% [markdown]
# Parse the custom arguments and set the output directory where the model will be saved: 

# %%
# Notebook cell 11
parser = transformers.HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(
    args=["--output_dir", "output"]
)

# %% [markdown]
# Setup the training dataset:

# %%
# Notebook cell 13
train_dataset = CustomDataset(args=args)

# %% [markdown]
# Check the shape of the dataset:

# %%
# Notebook cell 15
print("Input shape: ", train_dataset[0]['input_ids'].shape)

# %% [markdown]
# ## 4. Run the trainer and monitor the loss
# 
# First, set up a callback to log the loss values during training (note this cell is not shown in the video):

# %%
# Notebook cell 17
from transformers import Trainer, TrainingArguments, TrainerCallback

# Define a custom callback to log the loss values
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)

    def __init__(self):
        self.logs = []

# Initialize the callback
loss_logging_callback = LossLoggingCallback()

# %% [markdown]
# Then, create an instance of the Hugging Face `Trainer` object from the `transformers` library. Call the `train()` method of the trainder to initialize the training run:

# %%
# Notebook cell 19
from transformers import Trainer

trainer = Trainer(
    model=pretrained_model, 
    args=args, 
    train_dataset=train_dataset, 
    eval_dataset=None,
    callbacks=[loss_logging_callback] 
)

trainer.train()

# %% [markdown]
# You can use the code below to save intermediate model checkpoints in your own training run:

# %%
# Notebook cell 21
# Saving configuration
    # save_strategy: str = field(default="steps")          # Can be replaced with "epoch"
    # save_steps: int = field(default=3)                   # Frequency of saving training checkpoint
    # save_total_limit: int = field(default=2)             # The total number of checkpoints to be saved

# %% [markdown]
# ### Checking the performance of an intermediate checkpoint
# 
# Below, you can try generating text using an intermediate checkpoint of the model. This checkpoint was saved after 10,000 training steps. As you did in previous lessons, you'll use the Solar tokenizer and then set up a `TextStreater` object to display the text as it is generated: 

# %%
# Notebook cell 23
from transformers import AutoTokenizer, TextStreamer
model_name_or_path = "./models/TinySolar-248m-4k"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# %%
# Notebook cell 24
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import torch

model_name_or_path = "./models/upstage/output/checkpoint-10000"
model2 = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,    
)

# %%
# Notebook cell 25
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model2.device)

streamer = TextStreamer(
    tokenizer, 
    skip_prompt=True, 
    skip_special_tokens=True
)

outputs = model2.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True, 
    max_new_tokens=64,     
    do_sample=True,
    temperature=1.0,
)
