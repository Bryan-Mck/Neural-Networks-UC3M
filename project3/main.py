import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PART I - LOAD AND PREPROCESS DATA
# ============================================================

# Load data from CSV file
df = pd.read_csv("ETTh1.csv")
# Array of floats of the column "OT"
series = df["OT"].values.astype(np.float32)

# Normalize to zero mean and unit variance
mean = series.mean() # average
std  = series.std() # how spread the values are around mean
series = (series - mean) / std

# This normalization is to express the values in terms of how many
# standard deviations away from the mean a value is.
# This helps stabilize training and prevent large dominating values

# Subsample: keep 1 every 10 points
# select every 10th point to reduce computation
subsample_factor = 10
series_sub = series[::subsample_factor]
T_total = len(series_sub) # New series length

# ============================================================
# PART II - QUANTIZATION
# ============================================================

# This turns continuous variables into descrete variables

# 128 is chosen to be a middle ground for good training.
# Too little and multiple values flood into one bin from overcrowding
# Too many, and many values do not see any training
NUM_BINS = 128

# This converts a continuous var "x" into a bin index
# x_min and x_max give the range
def quantize_uniform(x, x_min, x_max):
    x_scaled = (x - x_min) / (x_max - x_min + 1e-8)
    bins = np.floor(NUM_BINS * x_scaled).astype(int)
    bins = np.clip(bins, 0, NUM_BINS - 1)
    return bins

# This converts a bin value into a continuous var "x"
def dequantize_uniform(bins, x_min, x_max):
    centers = (bins + 0.5) / NUM_BINS
    return x_min + centers * (x_max - x_min)


# ============================================================
# PART III - BUILD OVERLAPPING WINDOWS
# ============================================================

# Over lapping windows is dividing a sequence of tokens into chunks

T = 200   # sequence length
N = 5000  # number of windows

# Every window is a training example
# REMEMBER: t_total is length of subsampled series

rng = np.random.default_rng(0)
starts = rng.integers(0, T_total - (T + 1), size=N) # Randomly selectec starting points

# Quantize using full series range as placeholder
# will be recomputed from training data below
series_bins_full = quantize_uniform(
    series_sub, series_sub.min(), series_sub.max()
)

# s:s+T and s+1:s+T+1 are ranges
# series_bins_full is the subsampled array with all values quantized
# s is a value from starts
# X gets a whole sample starting from s with T sample size
# Y does the same thing but adds one shift to the right
# The shift helps turn this into a supervised learning task,
# Y is used as the correct answer for X, being the correct next word
X = np.stack([series_bins_full[s:s+T]     for s in starts], axis=0)
Y = np.stack([series_bins_full[s+1:s+T+1] for s in starts], axis=0)

# Train / test split
test_ratio = 0.2
N_total = X.shape[0] # Returns integer tuple
N_test = int(N_total * test_ratio)
N_train = N_total - N_test

perm  = np.random.permutation(N_total)
train_idx = perm[:N_train]
test_idx  = perm[N_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test,  Y_test  = X[test_idx],  Y[test_idx]

# Compute quantization bounds from training data only (no test leakage)
train_min = series_sub[
    np.concatenate([starts[train_idx] + i for i in range(T)])
].min()
train_max = series_sub[
    np.concatenate([starts[train_idx] + i for i in range(T)])
].max()

# Re-quantize using training bounds
X_train = quantize_uniform(
    dequantize_uniform(X_train, series_sub.min(), series_sub.max()),
    train_min, train_max
)
Y_train = quantize_uniform(
    dequantize_uniform(Y_train, series_sub.min(), series_sub.max()),
    train_min, train_max
)
X_test = quantize_uniform(
    dequantize_uniform(X_test, series_sub.min(), series_sub.max()),
    train_min, train_max
)
Y_test = quantize_uniform(
    dequantize_uniform(Y_test, series_sub.min(), series_sub.max()),
    train_min, train_max
)

print("Train:", X_train.shape, Y_train.shape)
print("Test :", X_test.shape,  Y_test.shape)


# ============================================================
# PART IV - DATASET
# ============================================================

class TokenSequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = TokenSequenceDataset(X_train, Y_train)
test_dataset  = TokenSequenceDataset(X_test,  Y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=64, shuffle=False)


# ============================================================
# PART V - MODELS
# ============================================================

# Three models made for comparison

# Base RNN
class DiscreteRNN(nn.Module):
    def __init__(self, num_bins=128, d_model=64, hidden_dim=32, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity="relu",
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        # x: (batch, seq_len) integer token IDs
        emb = self.embedding(x)   # (batch, seq_len, d_model)
        h, _ = self.rnn(emb)       # (batch, seq_len, hidden_dim)
        logits = self.output(h)      # (batch, seq_len, num_bins)
        return logits

# LSTN (long short term memory)
class DiscreteLSTM(nn.Module):
    def __init__(self, num_bins=128, d_model=64, hidden_dim=32, n_layers=1, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        emb = self.embedding(x)
        h, _ = self.lstm(emb)
        logits = self.output(h)
        return logits

# Transformer
class DiscreteTransformer(nn.Module):
    def __init__(self, num_bins=128, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding   = nn.Embedding(num_bins, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output      = nn.Linear(d_model, num_bins)

    def forward(self, x):
        emb = self.embedding(x)        # (batch, seq_len, d_model)
        emb = self.pos_encoder(emb)    # add positional info
        h = self.transformer(emb)    # (batch, seq_len, d_model)
        logits = self.output(h)           # (batch, seq_len, num_bins)
        return logits

# Since transformers use all tokens at the same time, postion
# encoding needs to be accessible to keep order and figure out the next token correctly
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# ============================================================
# PART VI - LOSS FUNCTION
# ============================================================

# Since this is now a discrete problem instead of continuous
# We use cross entropy to compare the probability distributions of the bins

def token_cross_entropy_loss(logits, targets):
    B, T, C = logits.shape
    logits  = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    return F.cross_entropy(logits, targets)


# ============================================================
# PART VII - TRAINING LOOP
# ============================================================

def train_model(model, train_loader, num_epochs=10, lr=1e-3, device="cpu"):
    model.to(device) # Sets to device 
    model.train() # Puts in training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = [] # The Ls

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = token_cross_entropy_loss(logits, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}  loss: {avg_loss:.4f}")

    return loss_history


# ============================================================
# PART VII b - TRAIN ALL MODELS
# ============================================================

# Train all models for comparison

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

rnn_model = DiscreteRNN(num_bins=128, d_model=64, hidden_dim=32, n_layers=1)
lstm_model = DiscreteLSTM(num_bins=128, d_model=64, hidden_dim=32, n_layers=2, drop_prob=0.3)
transformer_model = DiscreteTransformer(num_bins=128, d_model=64, nhead=4, num_layers=2)

print("\n--- Training RNN ---")
rnn_losses = train_model(rnn_model, train_loader, num_epochs=10, lr=1e-3, device=device)

print("\n--- Training LSTM ---")
lstm_losses = train_model(lstm_model, train_loader, num_epochs=10, lr=1e-3, device=device)

print("\n--- Training Transformer ---")
transformer_losses = train_model(transformer_model, train_loader, num_epochs=10, lr=1e-3, device=device)

# Plot training loss curves
plt.figure(figsize=(10, 4))
plt.plot(rnn_losses,         label="RNN")
plt.plot(lstm_losses,        label="LSTM")
plt.plot(transformer_losses, label="Transformer")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()




# ============================================================
# PART VIII - AUTOREGRESSIVE SAMPLING
# ============================================================

# @torch.no_grad() disables gradient tracking, and diables
# the ability to use backwards()
# Leads to less memory usage and faster outputs

@torch.no_grad()
def sample_tokens(model, seed_seq, steps=50, temperature=1.0, device="cpu"):
    model.eval() # Set to evaluation mode
    model.to(device) # mount device

    seq = seed_seq.clone().to(device)  # Starting seed sequence to predict

    for _ in range(steps): # Steps is how many tokens to predict in a row

        # Adds new dimension to seq
        x = seq.unsqueeze(0)            # (1, T)
        # Logits are any real number before being translated into probablities
        logits = model(x)                    # (1, T, num_bins)
        # Temperature controls the confience
        # If it should be flattened or have sharp peaks
        next_logits = logits[0, -1] / temperature
        # Softmax turns logits into probabilities
        probs = torch.softmax(next_logits, dim=-1)
        # Chooses a random value weighted from the probablities
        # Its ranodm and not always the highest
        next_token = torch.multinomial(probs, num_samples=1)
        # Concatenate the new token to the seqence
        seq = torch.cat([seq, next_token], dim=0)

    return seq.cpu() # To keep compatibility with numpy and other CPU operations

# This function turns indexes of a bin back into a value
# The value being the orignal continuous value
# REMEMBER: Tokens are indexes of the bins
def tokens_to_values(token_seq, x_min, x_max, num_bins=128):
    token_seq = np.asarray(token_seq)
    return dequantize_uniform(token_seq, x_min, x_max, num_bins=num_bins)


# ============================================================
# PART IX - EVALUATION
# ============================================================

T_train = 100          # tokens used as seed/context
H_eff = T - T_train  # forecasting horizon

# --------------------------------------------------------
# 1. ONE-STEP-AHEAD EVALUATION (teacher forcing)
# --------------------------------------------------------

# instead of giving the model the previous sequence of predictions
# We give it the actual correct sequence called teacher forcing
# This stops errors from piling up

@torch.no_grad()
def one_step_mse(model, test_loader, device="cpu"):
    model.eval() # Set to evaluation mode
    model.to(device)
    total_mse = 0.0
    count = 0

    # Bin_centers grabs a real value instead of the index 
    # using the range of train_min and train_max
    # Each value is a step of (train_max - train_min) apart / NUM_BINS - 1
    # Turing the probabilites of each bin into a continuous variable
    # This creates smoothing to avoid jumps
    bin_centers = torch.linspace(train_min, train_max, NUM_BINS).to(device)

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)

        # B is batch size, T is sequence length, 128 is number of bins
        logits = model(X_batch)                      
        probs = torch.softmax(logits, dim=-1)        
        pred_vals = (probs * bin_centers).sum(dim=-1)     

        # Convert labels back to real values
        # Y_batch is a tensor, converted to a numpy array
        true_vals = torch.tensor(
            # Converts from discrete to continuous
            # Not like linspace since it converts not creates
            dequantize_uniform(
                Y_batch.cpu().numpy(), train_min, train_max, NUM_BINS
            ),
            dtype=torch.float32
        ).to(device)

        # Stores the error buildup
        total_mse += F.mse_loss(pred_vals, true_vals).item()
        count += 1

    return total_mse / count # Average MSE


rnn_one_step_mse = one_step_mse(rnn_model, test_loader, device)
lstm_one_step_mse = one_step_mse(lstm_model, test_loader, device)
transformer_one_step_mse = one_step_mse(transformer_model, test_loader, device)

print(f"\nOne-step MSE  RNN:  {rnn_one_step_mse:.6f}")
print(f"One-step MSE  LSTM:  {lstm_one_step_mse:.6f}")
print(f"One-step MSE  Transformer: {transformer_one_step_mse:.6f}")


# --------------------------------------------------------
# 2. AUTOREGRESSIVE LONG-HORIZON FORECASTING
# --------------------------------------------------------

N_test_samples = X_test.shape[0]
forecast_rnn = np.zeros((N_test_samples, H_eff))
forecast_lstm = np.zeros((N_test_samples, H_eff))
forecast_transformer = np.zeros((N_test_samples, H_eff))

for i in range(N_test_samples):
    seed = torch.tensor(X_test[i, :T_train], dtype=torch.long)

    for model, forecast_arr in [
        (rnn_model, forecast_rnn),
        (lstm_model,   forecast_lstm),
        (transformer_model, forecast_transformer),
    ]:
        pred_tokens = sample_tokens(model, seed, steps=H_eff, device=device)
        pred_values = tokens_to_values(
            pred_tokens[T_train:].numpy(), train_min, train_max, NUM_BINS
        )
        forecast_arr[i] = pred_values


# --------------------------------------------------------
# 3. PLOT: one test sequence forecast vs ground truth
# --------------------------------------------------------

signal   = 10
t0       = max(0, T_train - 30)
true_vals = dequantize_uniform(X_test[signal], train_min, train_max, NUM_BINS)

plt.figure(figsize=(12, 5))
plt.plot(np.arange(t0, T), true_vals[t0:],                        'b.-', ms=6, label='Ground truth')
plt.plot(np.arange(T_train, T), forecast_rnn[signal],             'r-',        label='RNN forecast')
plt.plot(np.arange(T_train, T), forecast_lstm[signal],            'g-',        label='LSTM forecast')
plt.plot(np.arange(T_train, T), forecast_transformer[signal],     'm-',        label='Transformer forecast')
plt.axvline(x=T_train, color='k', linestyle='--', label='Forecasting begins')
plt.xlabel("Time step")
plt.ylabel("Normalized value")
plt.title(f"Autoregressive forecast on test sequence #{signal}")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------------
# 4. HORIZON MSE COMPARISON TABLE
# --------------------------------------------------------

true_horizon = dequantize_uniform(
    X_test[:, T_train:T_train + H_eff], train_min, train_max, NUM_BINS
)

mse_rnn         = np.mean((forecast_rnn         - true_horizon) ** 2)
mse_lstm        = np.mean((forecast_lstm        - true_horizon) ** 2)
mse_transformer = np.mean((forecast_transformer - true_horizon) ** 2)

print("\n--- Forecasting MSE over horizon ---")
print(f"RNN:         {mse_rnn:.6f}")
print(f"LSTM:        {mse_lstm:.6f}")
print(f"Transformer: {mse_transformer:.6f}")

results = pd.DataFrame({
    "Model":        ["RNN",              "LSTM",             "Transformer"],
    "One-step MSE": [rnn_one_step_mse,   lstm_one_step_mse,  transformer_one_step_mse],
    "Horizon MSE":  [mse_rnn,            mse_lstm,           mse_transformer],
})
print("\n", results.to_string(index=False))