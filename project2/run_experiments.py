import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset
from train_eval import train_standard, train_adversarial, evaluate
from calibration import calibration_summary
 
 
 
def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    quick: bool = False,
):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),              
    ])
    transform_test = T.Compose([
        T.ToTensor(),
    ])
 
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
 
    if quick:
        train_set = Subset(train_set, range(512))
        test_set  = Subset(test_set,  range(256))
 
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
 
 
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
 
 
def run_one_experiment(
    model_name: str,
    model_fn,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device: torch.device,
    out_dir: str,
    epsilon_train: float,
    num_epochs: int,
    eval_epsilons: list,
) -> dict:
    tag = f"{model_name}_eps{epsilon_train:.4f}"
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  ε_train : {epsilon_train:.4f}  ({epsilon_train*255:.1f}/255)")
    print(f"  Epochs  : {num_epochs}")
    print(f"{'='*60}")
 
    model = model_fn().to(device)
 
    # ── Train ──────────────────────────────────
    t0 = time.time()
    if epsilon_train == 0.0:
        history = train_standard(model, train_loader, device,
                                 num_epochs=num_epochs, verbose=True)
    else:
        history = train_adversarial(model, train_loader, device,
                                    epsilon=epsilon_train,
                                    num_epochs=num_epochs, verbose=True)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
 
    ckpt_path = os.path.join(out_dir, f"{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")
 
   
    results_per_eps = {}
    for eps_eval in eval_epsilons:
        print(f"\n  -- Eval at ε_eval={eps_eval:.4f} ({eps_eval*255:.1f}/255)")
        r = evaluate(model, test_loader, device,
                     epsilon_fgsm=eps_eval,
                     epsilon_pgd=eps_eval,
                     pgd_steps=10,
                     verbose=True)
        results_per_eps[eps_eval] = r
 

    rd_title = f"{model_name} {'AT ε='+str(round(epsilon_train*255))+'⁄255' if epsilon_train>0 else 'Standard'}"
    rd_path  = os.path.join(out_dir, f"reliability_{tag}.png")
    ece, nll, _ = calibration_summary(model, test_loader, device,
                                       title=rd_title, save_path=rd_path)
    print(f"\n  Reliability diagram saved → {rd_path}")
    print(f"  ECE={ece:.4f}  NLL={nll:.4f}")
 
    return {
        "model_name": model_name,
        "epsilon_train": epsilon_train,
        "history":  history,
        "train_time":  train_time,
        "results":   {str(k): v for k, v in results_per_eps.items()},
        "ece_clean":  ece,
        "nll_clean": nll,
    }
 
 
def make_sweep_plots(all_results: list, out_dir: str, model_name: str):
    res = [r for r in all_results if r["model_name"] == model_name]
    if not res:
        return
 
    epsilons   = np.array([r["epsilon_train"] for r in res])
    eps_labels = [f"{e*255:.0f}/255" for e in epsilons]
 
    def get_metrics(r):
        eps_key = str(r["epsilon_train"]) if r["epsilon_train"] > 0 else str(
            sorted(r["results"].keys(), key=float)[1])
        m = r["results"].get(eps_key) or list(r["results"].values())[0]
        return m
 
    clean_accs = [get_metrics(r)["clean_acc"] for r in res]
    fgsm_accs  = [get_metrics(r)["fgsm_acc"]  for r in res]
    pgd_accs   = [get_metrics(r)["pgd_acc"]   for r in res]
    eces  = [r["ece_clean"]               for r in res]
    nlls  = [r["nll_clean"]               for r in res]
 
    x = np.arange(len(epsilons))
    w = 0.25
 
    # ── Plot 1: Accuracy vs ε ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, clean_accs, width=w, label="Clean", color="steelblue")
    ax.bar(x,     fgsm_accs,  width=w, label="FGSM",  color="darkorange")
    ax.bar(x + w, pgd_accs,   width=w, label="PGD",   color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Training ε")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — Accuracy vs Training ε")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_accuracy_vs_eps.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
 
    # ── Plot 2: ECE vs ε ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons * 255, eces, marker="o", color="purple", linewidth=2)
    ax.set_xlabel("Training ε (×255)")
    ax.set_ylabel("ECE")
    ax.set_title(f"{model_name} — ECE vs Training ε")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_ece_vs_eps.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
 
    # ── Plot 3: NLL vs ε ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons * 255, nlls, marker="s", color="teal", linewidth=2)
    ax.set_xlabel("Training ε (×255)")
    ax.set_ylabel("NLL")
    ax.set_title(f"{model_name} — NLL vs Training ε")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_nll_vs_eps.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
 
    # ── Plot 4: Robust accuracy (FGSM & PGD) vs ε ─
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons * 255, fgsm_accs, marker="^", label="FGSM robust", color="darkorange")
    ax.plot(epsilons * 255, pgd_accs,  marker="v", label="PGD  robust", color="firebrick")
    ax.set_xlabel("Training ε (×255)")
    ax.set_ylabel("Robust Accuracy")
    ax.set_title(f"{model_name} — Robust Accuracy vs Training ε")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_robust_vs_eps.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
 
 
def make_comparison_plot(all_results: list, out_dir: str):
    models = list({r["model_name"] for r in all_results})
    if len(models) < 2:
        return
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, mname in zip(axes, sorted(models)):
        res = sorted([r for r in all_results if r["model_name"] == mname],
                     key=lambda r: r["epsilon_train"])
 
        eps_labels = [f"{r['epsilon_train']*255:.0f}/255" for r in res]
 
        def get_m(r, key):
            eps_key = str(r["epsilon_train"]) if r["epsilon_train"] > 0 else \
                      str(sorted(r["results"].keys(), key=float)[1])
            m = r["results"].get(eps_key) or list(r["results"].values())[0]
            return m[key]
 
        x = np.arange(len(res))
        w = 0.25
        ax.bar(x - w, [get_m(r, "clean_acc") for r in res], w,
               label="Clean", color="steelblue")
        ax.bar(x,     [get_m(r, "fgsm_acc")  for r in res], w,
               label="FGSM",  color="darkorange")
        ax.bar(x + w, [get_m(r, "pgd_acc")   for r in res], w,
               label="PGD",   color="firebrick")
        ax.set_title(mname)
        ax.set_xticks(x)
        ax.set_xticklabels(eps_labels)
        ax.set_xlabel("Training ε")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
 
    fig.suptitle("SmallCNN vs ResNet-18: Accuracy Breakdown", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "comparison_accuracy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
 