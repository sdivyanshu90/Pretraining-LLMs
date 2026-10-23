# %% [markdown]
# Source notebook: Lesson_4.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 4: Preparing your model for training

# %%
# Notebook cell 2
# Ignore insignificant warnings (ex: deprecation warnings)
import warnings
warnings.filterwarnings('ignore')

# Set a seed value for reproducibility
import torch

def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()

# %% [markdown]
# ## 1. Model configuration
# 
# You'll configure models based on Meta's Llama family of models. The transformers library has several tools for working with these models, which you can read about [here](https://huggingface.co/docs/transformers/main/en/model_doc/llama).
# 
# Start by creating a `LlamaConfig` object to configure the architecture of the model:

# %%
# Notebook cell 4
from transformers import LlamaConfig
config = LlamaConfig()
print(config)

# %% [markdown]
# Next, update parameters to change the model architecture:

# %%
# Notebook cell 6
config.num_hidden_layers = 12      # reduced from 32 to 12
config.hidden_size = 1024          # reduced 1/4 from 4096 to 1024
config.intermediate_size = 4096    # reduced 1/3 from 11008 to 4096 (dimension of MLP representations)
config.num_key_value_heads = 8     # reduced 1/4 from 32 to 8 (defaults to num_attention_heads=32)
config.torch_dtype = "bfloat16"    # for half-precision training
config.use_cache = False           # `True` is incompatible w/ gradient checkpointing
print(config)

# %% [markdown]
# ## 2. Weight initialization
# 
# In the next sections, you'll explore four different ways to initialize the weights of a model for training:
# 1. Random weight initialization
# 2. Using an existing model for continued pre-training
# 3. Downscaling an existing model
# 4. Upscaling an existing model

# %% [markdown]
# ### Random weight initialization
# 
# Randomly initializing model weights sets all weights to values from a truncated normal distribution with mean 0 and standard deviation of 0.02. Values beyond 2-sigma from the mean are set to 0.

# %%
# Notebook cell 9
from transformers import LlamaForCausalLM
model = LlamaForCausalLM(config)
print(model)

# %%
# Notebook cell 10
def print_nparams(model):
    """Calculate the total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")

print_nparams(model)  # 248013824 => 248M

# %% [markdown]
# Take a look at a sample of the weights in a single layer:

# %%
# Notebook cell 12
layer_name = "model.layers.0.self_attn.q_proj.weight"

for name, param in model.named_parameters():
    if name == layer_name:
        print(f"First 30 weights of layer '{layer_name}':")
        print(param.data.view(-1)[:30])
        break

# %% [markdown]
# Try using the model for inference:

# %%
# Notebook cell 14
# Load a tokenizer from Upstage Solar, 
# which is compatible with the Llama-2 tokenizer
from transformers import LlamaTokenizer
model_dir = "./models/SOLAR-10.7B-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# Run simple inference with prompt
from transformers import TextStreamer

prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(
    tokenizer, 
    skip_prompt=True, 
    skip_special_tokens=True
)

outputs = model.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True, 
    max_new_tokens=128, 
    do_sample=False
)

# %% [markdown]
# Remove the model from memory to avoid crashing the kernel:

# %%
# Notebook cell 16
# NOTE: We're running large models in a limited environment. Run me if you encounter any memory issues.
import gc
del model
del streamer
del outputs
gc.collect()

# %% [markdown]
# ### Reuse general pretrained model weights
# 
# If you load an existing model, you can use it as is to continue pretraining on new data.

# %%
# Notebook cell 18
from transformers import AutoModelForCausalLM

model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# %% [markdown]
# Remove the model from memory to avoid crashing the kernel:

# %%
# Notebook cell 20
# NOTE: We're running large models in a limited environment. Run me if you encounter any memory issues.
del model
gc.collect()

# %% [markdown]
# ### Downscaling from a general pretrained model
# 
# Here you'll downscale the tinySolar-248m-4k model from a 12 layer model to a 10 layer model.

# %%
# Notebook cell 22
from transformers import AutoTokenizer, AutoConfig

model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# %%
# Notebook cell 23
print(model)

# %%
# Notebook cell 24
print_nparams(model)  # 248013824 => 248M

# %% [markdown]
# Remove the middle two layers (layers 5 and 6) and update the configuration:

# %%
# Notebook cell 26
layers = model.model.layers
model.model.layers = layers[:5] + layers[-5:]

config = AutoConfig.from_pretrained(
    model_name_or_path,    
    num_hidden_layers=len(model.model.layers),
)
model.config = config

print_nparams(model)  # 217601024 => 217M

# %% [markdown]
# Clear the memory to avoid crashing the kernel:

# %%
# Notebook cell 28
# NOTE: We're running large models in a limited environment. Run me if you encounter any memory issues.
import gc
del model
gc.collect()

# %% [markdown]
# ### Depth Upscaling from a general pretrained model
# 
# Here you are going to upscale the tinySolar-248m-4k model from 12 layers to 16 layers. Here are the steps you'll take:
# 1. Configure a 16 layer model and initialize it with random weights
# 2. Load the 12 layer tinySolar-248m-4k model into memory
# 3. Copy the bottom 8 and top 8 layers from the 12 layer model and use them to overwrite the random weights of the 16 layer model
# 4. Copy over the embedding and classifying layers to replace the randomly initialized counterparts in the 16 layer model

# %%
# Notebook cell 30
config = LlamaConfig(
    num_hidden_layers=16,  # We want our model to have 16 final layers
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    torch_dtype="bfloat16",
    use_cache=False 
)
print(config)

# %%
# Notebook cell 31
model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)  # convert to bfloat16
print_nparams(model)  # 308839424 => 308M

# %%
# Notebook cell 32
model_name_or_path = "upstage/TinySolar-248m-4k"
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,    
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print_nparams(pretrained_model) #  248013824 => 248M

# %%
# Notebook cell 33
from copy import deepcopy

model.model.layers = deepcopy(pretrained_model.model.layers[:-4]) \
    + deepcopy(pretrained_model.model.layers[4:])

model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)

model.lm_head = deepcopy(pretrained_model.lm_head)

print(model.config)

# %% [markdown]
# Check the number of parameters is still 308 million:

# %%
# Notebook cell 35
print_nparams(model)  # 308839424 => 308M

# %% [markdown]
# Try using the model for inference:

# %%
# Notebook cell 37
# Run simple inference to show no trained model
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(
    tokenizer, 
    skip_prompt=True, 
    skip_special_tokens=True
)

outputs = model.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True, 
    max_new_tokens=128, 
    do_sample=False
)

# %% [markdown]
# ### Save the model to disk
# 
# Note the new model name here which reflects the 308 million parameters of the new, upscaled model. 

# %%
# Notebook cell 39
model.save_pretrained('./data/TinySolar-308m-4k-init')

# %%
# Notebook cell 40


# %%
# Notebook cell 41


# %%
# Notebook cell 42


# %%
# Notebook cell 43


# %%
# Notebook cell 44


# %%
# Notebook cell 45


# %%
# Notebook cell 46


# %%
# Notebook cell 47


# %%
# Notebook cell 48


# %%
# Notebook cell 49


# %%
# Notebook cell 50


# %%
# Notebook cell 51


# %%
# Notebook cell 52


# %%
# Notebook cell 53


# %%
# Notebook cell 54


# %%
# Notebook cell 55

