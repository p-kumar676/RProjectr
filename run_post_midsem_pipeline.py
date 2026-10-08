import argparse
import subprocess
import sys
import os

def run_command(cmd, description):
    print("\n" + "="*60)
    print(f" [RUNNING] {description}")
    print("="*60)
    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[!] Error executing step: {description}")
        sys.exit(result.returncode)
    print(f"\n[OK] Step Completed: {description}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Child DINOv2 Feature Distillation Pipeline")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'stl10'], help='Dataset for training & benchmarking')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs per training step')
    args = parser.parse_args()

    print("#"*60)
    print(f" DINOv2 MULTI-CHILD DISTILLATION PIPELINE (MobileNetV3 + ResNet-18) ")
    print(f" Dataset Target: {args.dataset.upper()}")
    print("#"*60)

    python_exe = sys.executable
    ds = args.dataset
    ep = str(args.epochs)

    # 1. Pipeline Verification
    run_command(
        [python_exe, "src/verify_pipeline.py"],
        "1. Pipeline & Multi-Child Architecture Verification"
    )

    # 2. Baseline Training for Both Child Models
    run_command(
        [python_exe, "src/train_baseline.py", "--model", "mobilenet", "--dataset", ds, "--epochs", ep, "--save_path", f"baseline_mobilenet_{ds}.pth"],
        f"2a. Supervised Baseline Training: MobileNetV3-Small ({ds.upper()})"
    )
    run_command(
        [python_exe, "src/train_baseline.py", "--model", "resnet18", "--dataset", ds, "--epochs", ep, "--save_path", f"baseline_resnet18_{ds}.pth"],
        f"2b. Supervised Baseline Training: ResNet-18 ({ds.upper()})"
    )

    # 3. Distillation Training from DINOv2 Teacher for Both Child Models
    run_command(
        [python_exe, "src/train_distill.py", "--model", "mobilenet", "--dataset", ds, "--epochs", ep, "--alpha", "0.5", "--save_path", f"distilled_mobilenet_{ds}.pth"],
        f"3a. DINOv2 Feature Distillation: MobileNetV3-Small ({ds.upper()})"
    )
    run_command(
        [python_exe, "src/train_distill.py", "--model", "resnet18", "--dataset", ds, "--epochs", ep, "--alpha", "0.5", "--save_path", f"distilled_resnet18_{ds}.pth"],
        f"3b. DINOv2 Feature Distillation: ResNet-18 ({ds.upper()})"
    )

    # 4. Hyperparameter Sweeps across Alpha values for Both Child Models
    run_command(
        [python_exe, "src/sweep_hyperparameters.py", "--model", "mobilenet", "--dataset", ds, "--epochs", ep, "--output_json", f"sweep_mobilenet_{ds}.json", "--plot_out", f"sweep_mobilenet_{ds}.png"],
        f"4a. Hyperparameter Sweep: MobileNetV3 ({ds.upper()})"
    )
    run_command(
        [python_exe, "src/sweep_hyperparameters.py", "--model", "resnet18", "--dataset", ds, "--epochs", ep, "--output_json", f"sweep_resnet18_{ds}.json", "--plot_out", f"sweep_resnet18_{ds}.png"],
        f"4b. Hyperparameter Sweep: ResNet-18 ({ds.upper()})"
    )

    # 5. Comprehensive Evaluation, Latency, t-SNE & Edge Export (ONNX + TorchScript)
    run_command(
        [python_exe, "src/evaluate.py", "--dataset", ds, "--baseline_mobilenet_ckpt", f"baseline_mobilenet_{ds}.pth", "--distilled_mobilenet_ckpt", f"distilled_mobilenet_{ds}.pth", "--baseline_resnet18_ckpt", f"baseline_resnet18_{ds}.pth", "--distilled_resnet18_ckpt", f"distilled_resnet18_{ds}.pth", "--export_onnx"],
        f"5. Edge Latency Benchmarking, t-SNE Plot & Model Export ({ds.upper()})"
    )

    print("\n" + "#"*60)
    print(" MULTI-CHILD DISTILLATION PIPELINE COMPLETED SUCCESSFULLY! ")
    print("#"*60)

if __name__ == '__main__':
    main()
