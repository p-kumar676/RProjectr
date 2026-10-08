import json
import os

def build_complete_colab_notebook():
    # Read files from src/
    def read_src_file(filename):
        path = os.path.join('src', filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    dataset_code = read_src_file('dataset.py')
    models_code = read_src_file('models.py')
    losses_code = read_src_file('losses.py')
    utils_code = read_src_file('utils.py')
    verify_code = read_src_file('verify_pipeline.py')
    train_base_code = read_src_file('train_baseline.py')
    train_dist_code = read_src_file('train_distill.py')
    sweep_code = read_src_file('sweep_hyperparameters.py')
    eval_code = read_src_file('evaluate.py')

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell_header",
                "metadata": {},
                "source": [
                    "# 🧠 DINOv2 → Multi-Child Cross-Architecture Feature Distillation\n",
                    "**Full Complete Pipeline | Google Colab (T4 GPU)**\n",
                    "\n",
                    "---\n",
                    "**Features Included:**\n",
                    "- **Teacher Model**: DINOv2 (ViT-S/14, 384d CLS embedding)\n",
                    "- **2 Child Models**: MobileNetV3-Small (~2.5M params) & ResNet-18 (~11.2M params)\n",
                    "- **2 Datasets**: CIFAR-100 & STL-10 (High-Res 96x96 SSL Benchmark)\n",
                    "- **5 Research Plots**: Pareto Frontier, t-SNE, Multi-Dataset Comparison, Cosine Heatmap, Bubble Chart\n",
                    "- **Hyperparameter Sweeps**: Loss balance weighting (α ∈ {0.1, 0.5, 1.0})\n",
                    "- **Edge Export**: ONNX & TorchScript formats\n",
                    "\n",
                    "> ⚠️ **Make sure you have selected Runtime → Change runtime type → T4 GPU before running!**"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec0",
                "metadata": {},
                "source": ["## ⚙️ Cell 0 — Install Dependencies & GPU Environment Check"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_env_check",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install all required packages\n",
                    "%pip install -q torch torchvision scikit-learn matplotlib seaborn tqdm onnx onnxruntime\n",
                    "\n",
                    "import torch\n",
                    "import sys\n",
                    "\n",
                    "print('=' * 60)\n",
                    "print(' ENVIRONMENT CHECK')\n",
                    "print('=' * 60)\n",
                    "print(f'Python version : {sys.version}')\n",
                    "print(f'PyTorch version: {torch.__version__}')\n",
                    "\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'✅ GPU         : {torch.cuda.get_device_name(0)}')\n",
                    "    print(f'   CUDA version: {torch.version.cuda}')\n",
                    "    print(f'   VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
                    "    DEVICE = torch.device('cuda')\n",
                    "else:\n",
                    "    print('⚠️  No GPU detected — running on CPU (training will be slow!)')\n",
                    "    print('   Go to Runtime → Change runtime type → T4 GPU')\n",
                    "    DEVICE = torch.device('cpu')\n",
                    "\n",
                    "print(f'\\nUsing device: {DEVICE}')"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec1",
                "metadata": {},
                "source": [
                    "## 💾 Cell 1 — Recreate Complete Source Modules (Self-Contained Setup)\n",
                    "Creates all `src/` python modules inline so the notebook runs standalone even without Drive mounting."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_recreate_src",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "os.makedirs('src', exist_ok=True)\n",
                    "os.makedirs('data', exist_ok=True)\n",
                    "\n",
                    "def write_file(path, content):\n",
                    "    with open(path, 'w', encoding='utf-8') as f:\n",
                    "        f.write(content)\n",
                    "    print(f'  [OK] Created {path}')\n",
                    "\n",
                    "print('[+] Writing source modules to src/ directory...')\n",
                    "write_file('src/dataset.py', '''" + dataset_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/models.py', '''" + models_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/losses.py', '''" + losses_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/utils.py', '''" + utils_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/verify_pipeline.py', '''" + verify_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/train_baseline.py', '''" + train_base_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/train_distill.py', '''" + train_dist_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/sweep_hyperparameters.py', '''" + sweep_code.replace("'", "\\'") + "''')\n",
                    "write_file('src/evaluate.py', '''" + eval_code.replace("'", "\\'") + "''')\n",
                    "\n",
                    "print('✅ All source code modules successfully created inside src/!')"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec2",
                "metadata": {},
                "source": ["## 🧪 Cell 2 — Verify Multi-Child Architecture Pipeline"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_verify_pipeline",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.append('src')\n",
                    "from verify_pipeline import verify_pipeline\n",
                    "\n",
                    "verify_pipeline()"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec3",
                "metadata": {},
                "source": [
                    "## 🏋️ Cell 3 — Train Child Model 1: MobileNetV3-Small (CIFAR-100 & STL-10)\n",
                    "Trains both baseline supervised and DINOv2 feature distilled MobileNetV3-Small."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_train_mobilenet",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train Baseline Supervised MobileNetV3 (CIFAR-100)\n",
                    "!python src/train_baseline.py --model mobilenet --dataset cifar100 --epochs 15 --save_path baseline_mobilenet.pth\n",
                    "\n",
                    "# Train DINOv2 Feature Distilled MobileNetV3 (CIFAR-100)\n",
                    "!python src/train_distill.py --model mobilenet --dataset cifar100 --epochs 15 --alpha 0.5 --save_path distilled_mobilenet.pth\n",
                    "\n",
                    "# (Optional) Train MobileNetV3 on STL-10 (96x96 High-Res SSL Benchmark)\n",
                    "# !python src/train_baseline.py --model mobilenet --dataset stl10 --epochs 15 --save_path baseline_mobilenet_stl10.pth\n",
                    "# !python src/train_distill.py --model mobilenet --dataset stl10 --epochs 15 --alpha 0.5 --save_path distilled_mobilenet_stl10.pth"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec4",
                "metadata": {},
                "source": [
                    "## 🏋️ Cell 4 — Train Child Model 2: ResNet-18 (CIFAR-100 & STL-10)\n",
                    "Trains both baseline supervised and DINOv2 feature distilled ResNet-18."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_train_resnet",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train Baseline Supervised ResNet-18 (CIFAR-100)\n",
                    "!python src/train_baseline.py --model resnet18 --dataset cifar100 --epochs 15 --save_path baseline_resnet18.pth\n",
                    "\n",
                    "# Train DINOv2 Feature Distilled ResNet-18 (CIFAR-100)\n",
                    "!python src/train_distill.py --model resnet18 --dataset cifar100 --epochs 15 --alpha 0.5 --save_path distilled_resnet18.pth\n",
                    "\n",
                    "# (Optional) Train ResNet-18 on STL-10 (96x96 High-Res SSL Benchmark)\n",
                    "# !python src/train_baseline.py --model resnet18 --dataset stl10 --epochs 15 --save_path baseline_resnet18_stl10.pth\n",
                    "# !python src/train_distill.py --model resnet18 --dataset stl10 --epochs 15 --alpha 0.5 --save_path distilled_resnet18_stl10.pth"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec5",
                "metadata": {},
                "source": ["## 🎛️ Cell 5 — Hyperparameter Sweeps (Alpha Loss Weighting Trade-Off)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_hyperparameter_sweep",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run hyperparameter sweeps for α ∈ {0.1, 0.5, 1.0}\n",
                    "!python src/sweep_hyperparameters.py --model mobilenet --dataset cifar100 --epochs 3 --output_json sweep_mobilenet.json --plot_out sweep_mobilenet.png\n",
                    "!python src/sweep_hyperparameters.py --model resnet18 --dataset cifar100 --epochs 3 --output_json sweep_resnet18.json --plot_out sweep_resnet18.png"
                ]
            },
            {
                "cell_type": "markdown",
                "id": "cell_sec6",
                "metadata": {},
                "source": ["## 📊 Cell 6 — Comprehensive Evaluation & 5 Research Visualization Plots"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_eval_plots",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run evaluation, latency profiling, and export ONNX / TorchScript models\n",
                    "!python src/evaluate.py --dataset cifar100 --export_onnx\n",
                    "\n",
                    "import matplotlib.pyplot as plt\n",
                    "from PIL import Image\n",
                    "import os\n",
                    "\n",
                    "fig_names = [\n",
                    "    'pareto_efficiency_frontier.png',\n",
                    "    'tsne_comparison.png',\n",
                    "    'multi_dataset_comparison.png',\n",
                    "    'cosine_similarity_heatmap.png',\n",
                    "    'efficiency_bubble_chart.png'\n",
                    "]\n",
                    "\n",
                    "for fig_name in fig_names:\n",
                    "    if os.path.exists(fig_name):\n",
                    "        print(f'\\n🖼️ Displaying: {fig_name}')\n",
                    "        img = Image.open(fig_name)\n",
                    "        plt.figure(figsize=(10, 6))\n",
                    "        plt.imshow(img)\n",
                    "        plt.axis('off')\n",
                    "        plt.show()"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('newDINOv2_Distillation_Colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Successfully built 100% complete self-contained notebook: newDINOv2_Distillation_Colab.ipynb ({os.path.getsize('newDINOv2_Distillation_Colab.ipynb') / 1024:.1f} KB)")

if __name__ == '__main__':
    build_complete_colab_notebook()
