import os
import torch

from run_experiments import (
    get_dataloaders,
    run_one_experiment,
    make_sweep_plots,
    make_comparison_plot,
)

from models import SmallCNN, get_resnet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    num_epochs = 30

    train_epsilons = [0.0, 2/255, 4/255, 8/255, 16/255]
    eval_epsilons  = train_epsilons

    train_loader, test_loader = get_dataloaders(
        data_dir="./data",
        batch_size=128,
        quick=False
    )

    all_results = []

    print("small cnn:")
    for eps in train_epsilons:
        result = run_one_experiment(
            model_name="SmallCNN",
            model_fn=SmallCNN,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            out_dir=out_dir,
            epsilon_train=eps,
            num_epochs=num_epochs,
            eval_epsilons=eval_epsilons,
        )
        all_results.append(result)

    make_sweep_plots(all_results, out_dir, "SmallCNN")

    print("restnet")
    for eps in train_epsilons:
        result = run_one_experiment(
            model_name="ResNet18",
            model_fn=get_resnet18,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            out_dir=out_dir,
            epsilon_train=eps,
            num_epochs=num_epochs,
            eval_epsilons=eval_epsilons,
        )
        all_results.append(result)

    make_sweep_plots(all_results, out_dir, "ResNet18")

    print("\ncomparison:")

    make_comparison_plot(all_results, out_dir)

    print("\ndone")


if __name__ == "__main__":
    main()

    