import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt

from typing import Tuple
 
 # Get the expected calibration error
def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    # This creates equally spaced bin edges
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
 
    # Low is the lower bound of a bin and high is the upper
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        # Check if empty bin
        if mask.sum() == 0:
            continue
        bin_acc  = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        # Total num of data points divied by labels, 
        # Multiplied by the gap between accuracy and confidence
        ece += (mask.sum() / n) * (abs(bin_acc - bin_conf))
 
    return float(ece)
 
# Compute the negative log-likelihood (NLL) 
# Used for labels
def compute_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_labels = torch.tensor(labels, dtype=torch.long)
    nll = F.cross_entropy(t_logits, t_labels).item()
    return float(nll)
 
 
# summaries 

def reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: str = None,
) -> plt.Figure:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs   = []
    bin_confs  = []
    bin_counts = []
 
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append((predictions[mask] == labels[mask]).mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
 
    bin_accs  = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_mids, bin_accs, width=1.0 / n_bins, alpha=0.7,
           edgecolor="black", label="Model accuracy", color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration")
 
    # Shade the gap (miscalibration)
    for mid, acc in zip(bin_mids, bin_accs):
        ax.bar(mid, mid - acc, bottom=acc, width=1.0 / n_bins,
               alpha=0.25, color="red", edgecolor="none")
 
    ece = compute_ece(confidences, predictions, labels, n_bins)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
 
    return fig
 

def calibration_summary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    title: str = "Reliability Diagram",
    save_path: str = None,
) -> Tuple[float, float, plt.Figure]:
    model.eval()
    all_logits = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
 
    all_logits = torch.cat(all_logits).numpy() 
    all_labels = torch.cat(all_labels).numpy() 
 
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    confs = probs.max(axis=1)            
    preds = probs.argmax(axis=1)         
 
    ece = compute_ece(confs, preds, all_labels)
    nll = compute_nll(all_logits, all_labels)
    fig = reliability_diagram(confs, preds, all_labels, title=title, save_path=save_path)
 
    return ece, nll, fig
 
 