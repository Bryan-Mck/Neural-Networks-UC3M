import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from calibration import compute_ece, compute_nll
 

 # ATTACKS ---

 # Fast gradient sign method attack
def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
 
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward() # writes to .grad
 
    with torch.no_grad():
        perturbation = epsilon * images.grad.sign()
        adv_images   = torch.clamp(images + perturbation, 0.0, 1.0)
 
    return adv_images.detach()
 
 
 # Projected Gradient desecent attack
def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float  = None,
    num_steps: int = 10,
    criterion: nn.Module = None,
    random_start: bool = True,
) -> torch.Tensor:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if alpha is None:
        alpha = epsilon / 4.0
 
    adv_images = images.clone().detach()
 
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
 
    for _ in range(num_steps):
        adv_images = adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
 
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
 
    return adv_images
 

 # Training and Eval ---

# Standard training loop for CNN
def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_type: str = "cosine",
    verbose: bool = True,
) -> list:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
 
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
 
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
 
        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  [Standard] Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | {elapsed:.1f}s")
 
    return history
 
 # Training with the adversarial examples
def train_adversarial(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epsilon: float = 8 / 255,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_type: str = "cosine",
    verbose: bool = True,
) -> list:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
 
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
 
            # Generate adversarial examples (detached — no graph leak)
            adv_images = fgsm_attack(model, images, labels, epsilon, criterion)
 
            # Train on adversarial examples
            optimizer.zero_grad()
            loss = criterion(model(adv_images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
 
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
 
        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  [AdvTrain ε={epsilon:.4f}] Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | {elapsed:.1f}s")
 
    return history
 
 
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon_fgsm: float = 8 / 255,
    epsilon_pgd:  float = 8 / 255,
    pgd_steps:    int   = 10,
    verbose:      bool  = True,
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
 
    clean_correct  = 0
    fgsm_correct   = 0
    pgd_correct    = 0
    total          = 0
 
    all_logits_clean = []
    all_labels_list  = []
 
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        bs = images.size(0)
        total += bs
 
        # Class
        with torch.no_grad():
            logits_clean = model(images)
            preds_clean  = logits_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()
            all_logits_clean.append(logits_clean.cpu())
            all_labels_list.append(labels.cpu())
 
        # FGSM
        adv_fgsm = fgsm_attack(model, images, labels, epsilon_fgsm, criterion)
        with torch.no_grad():
            preds_fgsm = model(adv_fgsm).argmax(dim=1)
            fgsm_correct += (preds_fgsm == labels).sum().item()
 
        # PGD
        adv_pgd = pgd_attack(model, images, labels, epsilon_pgd,
                              num_steps=pgd_steps)
        with torch.no_grad():
            preds_pgd = model(adv_pgd).argmax(dim=1)
            pgd_correct += (preds_pgd == labels).sum().item()
 
    # ── Calibration metrics ───────────────────
    all_logits = torch.cat(all_logits_clean).numpy()
    all_labels = torch.cat(all_labels_list).numpy()
 
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
 
    ece = compute_ece(confs, preds, all_labels)
    nll = compute_nll(all_logits, all_labels)
 
    results = {
        "clean_acc":  clean_correct  / total,
        "fgsm_acc":   fgsm_correct   / total,
        "pgd_acc":    pgd_correct     / total,
        "ece":        ece,
        "nll":        nll,
    }
 
    if verbose:
        print(f"  Clean acc  : {results['clean_acc']:.4f}")
        print(f"  FGSM  acc  : {results['fgsm_acc']:.4f}  (ε={epsilon_fgsm:.4f})")
        print(f"  PGD   acc  : {results['pgd_acc']:.4f}   (ε={epsilon_pgd:.4f}, {pgd_steps} steps)")
        print(f"  ECE        : {results['ece']:.4f}")
        print(f"  NLL        : {results['nll']:.4f}")
 
    return results
 