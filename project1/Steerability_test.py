import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def __init__():
    pass

# Concept names matching the 8 columns in your concept_layer output
CONCEPT_NAMES = [
    "has_heel",       # 0
    "has_laces",      # 1
    "has_sleeve",     # 2
    "is_legwear",     # 3
    "is_footwear",    # 4
    "has_strap",      # 5
    "is_outerwear",   # 6
    "is_accessory",   # 7
]


def run_steerability_experiment(model, test_loader, device, num_batches=20):
    """
    For each concept k, flip its predicted value (0->1 or 1->0) and measure
    how much the final label prediction changes.

    Returns a dict with per-concept results:
        - mean_prob_change:  average absolute change in predicted class probability
        - flip_rate:         fraction of samples where predicted class changes
    """
    model.eval()
    num_concepts = len(CONCEPT_NAMES)

    # Accumulators - one entry per concept
    total_prob_change = torch.zeros(num_concepts)   # sum of |p_after - p_before|
    total_class_flips = torch.zeros(num_concepts)   # count of samples that changed class
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            # STEP 1: Normal forward pass
            # Get the concept predictions and original logits
            concept_preds, fc_logits, sc_logits = model(images)
            # concept_preds shape: [batch, 8] - values between 0 and 1

            # Original combined prediction (f(c) + s(x))
            original_logits = fc_logits + sc_logits                 # [batch, 10]
            original_probs  = F.softmax(original_logits, dim=1)     # [batch, 10]
            original_class  = original_logits.argmax(dim=1)         # [batch]

            # The probability the model assigned to its own top prediction
            original_top_prob = original_probs[
                torch.arange(batch_size), original_class
            ]

            # STEP 2: Intervene on each concept one at a time
            for k in range(num_concepts):

                # Flip concept k: copy the concept vector, then replace column k
                intervened_concepts = concept_preds.clone()          # [batch, 8]
                intervened_concepts[:, k] = 1.0 - concept_preds[:, k]
                # e.g. if concept_preds[:,k] = 0.85  ->  intervened = 0.15
                # This tells the model "actually this concept is absent/present"

                # Recompute ONLY the concept-based logits with the new concepts.
                # We keep the side-channel logits the same because s(x) doesn't
                # depend on concepts - it reads raw features directly.
                intervened_fc_logits = model.decision_layer(intervened_concepts)  # [batch, 10]

                # Combine with the unchanged side channel
                intervened_logits = intervened_fc_logits + sc_logits             # [batch, 10]
                intervened_probs  = F.softmax(intervened_logits, dim=1)          # [batch, 10]
                intervened_class  = intervened_logits.argmax(dim=1)              # [batch]

                # METRIC 1: average absolute change in top-class probability
                # Compare the probability of the *original* predicted class
                # before and after the intervention
                intervened_top_prob = intervened_probs[
                    torch.arange(batch_size), original_class
                ]
                prob_change = (intervened_top_prob - original_top_prob).abs()   # [batch]
                total_prob_change[k] += prob_change.sum().item()

                # METRIC 2: did the predicted class actually change?
                class_changed = (intervened_class != original_class).float()    # [batch]
                total_class_flips[k] += class_changed.sum().item()

    # Aggregate over all samples
    mean_prob_change = (total_prob_change / total_samples).numpy()
    flip_rate        = (total_class_flips  / total_samples).numpy()

    return {
        "mean_prob_change": mean_prob_change,
        "flip_rate":        flip_rate,
    }


def plot_steerability(results, title="Concept Intervention Effects"):
    """
    Two side-by-side bar charts:
        Left  - average change in predicted class probability per concept
        Right - fraction of samples that changed predicted class per concept
    """
    names      = CONCEPT_NAMES
    prob_delta = results["mean_prob_change"]
    flip_rate  = results["flip_rate"]

    # Sort concepts by average probability change (most influential first)
    order = np.argsort(prob_delta)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Left: mean probability change
    ax = axes[0]
    bars = ax.barh(
        [names[i] for i in order],
        prob_delta[order],
        color="steelblue"
    )
    ax.set_xlabel("Mean |delta probability| after intervention")
    ax.set_title("Average probability change")
    ax.invert_yaxis()   # highest at top

    for bar, val in zip(bars, prob_delta[order]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # Right: class flip rate
    ax = axes[1]
    bars = ax.barh(
        [names[i] for i in order],
        flip_rate[order],
        color="coral"
    )
    ax.set_xlabel("Fraction of samples that changed predicted class")
    ax.set_title("Class flip rate")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    for bar, val in zip(bars, flip_rate[order]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("steerability_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved -> steerability_results.png")


def print_concept_ranking(results):
    """Print a ranked table of concepts by influence."""
    prob_delta = results["mean_prob_change"]
    flip_rate  = results["flip_rate"]
    order      = np.argsort(prob_delta)[::-1]

    print("\n-- Concept Ranking by Intervention Effect --")
    print(f"{'Rank':<5} {'Concept':<20} {'Mean |dP|':<15} {'Flip Rate':<12}")
    print("-" * 55)
    for rank, idx in enumerate(order, 1):
        print(f"{rank:<5} {CONCEPT_NAMES[idx]:<20} "
              f"{prob_delta[idx]:<15.4f} {flip_rate[idx]:<12.4f}")


# HOW TO CALL THIS FROM YOUR MAIN SCRIPT
#
#   After you have trained your model and have test_loader ready:
#
#   from steerability import run_steerability_experiment, plot_steerability, print_concept_ranking
#
#   results = run_steerability_experiment(model, test_loader, device, num_batches=50)
#   print_concept_ranking(results)
#   plot_steerability(results, title="CBM -- Concept Intervention Effects")