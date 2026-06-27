# --- Configuration ---

import torch

# Define vocabulary size and transformer configuration
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 512        # Maximum sequence length for the model
N_EMBED = 2048              # Dimension of the embedding space
N_HEAD = 16                 # Number of attention heads in each transformer block
N_BLOCKS = 64               # Number of transformer blocks in the model

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"  # File path for the training dataset
DEV_PATH = "data/val/pile_dev.h5"      # File path for the validation dataset

# Transformer training parameters
T_BATCH_SIZE = 32          # Number of samples per training batch
T_CONTEXT_LENGTH = 16      # Context length for training batches
T_TRAIN_STEPS = 200000     # Total number of training steps
T_EVAL_STEPS = 1000        # Frequency (in steps) to perform evaluation
T_EVAL_ITERS = 250         # Number of iterations to evaluate the model
T_LR_DECAY_STEP = 50000    # Step at which to decay the learning rate (legacy, used if no warmup)
T_LR = 5e-4                # Initial (maximum) learning rate for training
T_LR_DECAYED = 5e-5        # Minimum learning rate
T_OUT_PATH = "models/transformer_B.pt"  # Path to save the trained model
T_CHECKPOINT_STEPS = 0      # Save periodic checkpoints every N steps (0 disables)
T_KEEP_LAST_CHECKPOINTS = 3 # Number of periodic checkpoints to keep (0 keeps all)
T_CHECKPOINT_DIR = None     # Optional checkpoint directory override

# Learning rate schedule (warmup + cosine decay)
T_WARMUP_STEPS = 2000       # Number of warmup steps (linear warmup from 0 to max LR)
T_USE_WARMUP = True         # Enable warmup + cosine decay schedule
T_SCHEDULE_TYPE = 'cosine'  # LR schedule type: 'cosine' or 'step'

# Model architecture options
DROPOUT = 0.0               # Dropout probability (0 = disabled). Recommended: 0.1 for small models
NORM_TYPE = 'layernorm'     # Normalization type: 'layernorm' or 'rmsnorm'
ACTIVATION = 'gelu'         # MLP activation: 'gelu' (GPT-2) or 'swiglu' (Llama)
TIE_WEIGHTS = True          # Tie token embedding and lm_head weights

# Memory-optimisation knobs (all OFF by default => unchanged behaviour/numerics).
USE_AMP = False                 # bf16/fp16 autocast (CUDA only; ignored on CPU)
AMP_DTYPE = "bf16"              # "bf16" (no GradScaler) or "fp16" (GradScaler)
USE_GRADIENT_CHECKPOINTING = False  # recompute block activations in backward to save VRAM
GRAD_ACCUM_STEPS = 1           # micro-batches per optimizer step (effective batch x N)
REPORT_MEMORY_BUDGET = False   # print a rough VRAM budget before training (CUDA only)

# torch.compile (PyTorch 2.0+)
USE_TORCH_COMPILE = False      # Enable torch.compile for 1.5-2x speedup (PyTorch 2.0+ only)

# Signal handling for graceful checkpoint on interrupt
ENABLE_SIGNAL_HANDLER = True   # Save checkpoint on SIGINT/SIGTERM

# Weight decay differentiation (don't apply to biases and LayerNorm)
WEIGHT_DECAY = 0.1             # Weight decay for AdamW
DIFFERENTIATED_WEIGHT_DECAY = True  # Don't apply weight decay to biases and norms

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Store all configurations in a dictionary for easy access and modification
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    't_checkpoint_steps': T_CHECKPOINT_STEPS,
    't_keep_last_checkpoints': T_KEEP_LAST_CHECKPOINTS,
    't_checkpoint_dir': T_CHECKPOINT_DIR,
    # New parameters
    't_warmup_steps': T_WARMUP_STEPS,
    't_use_warmup': T_USE_WARMUP,
    't_schedule_type': T_SCHEDULE_TYPE,
    'dropout': DROPOUT,
    'norm_type': NORM_TYPE,
    'activation': ACTIVATION,
    'tie_weights': TIE_WEIGHTS,
    'use_amp': USE_AMP,
    'amp_dtype': AMP_DTYPE,
    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
    'grad_accum_steps': GRAD_ACCUM_STEPS,
    'report_memory_budget': REPORT_MEMORY_BUDGET,
    'use_torch_compile': USE_TORCH_COMPILE,
    'enable_signal_handler': ENABLE_SIGNAL_HANDLER,
    'weight_decay': WEIGHT_DECAY,
    'differentiated_weight_decay': DIFFERENTIATED_WEIGHT_DECAY,
    'device': DEVICE,
}
