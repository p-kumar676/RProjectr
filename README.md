# Cross-Architecture Knowledge Distillation: DINOv2 → MobileNetV3

This repository contains the complete implementation and academic project blueprint for transferring self-supervised representations from **Meta's DINOv2 (ViT-S/14)** to **MobileNetV3-Small** for edge-efficient vision tasks.

---

## 📌 Project Overview & Structure

```
dinov2_mobilenet_distillation/
├── README.md                          # Main project guide
├── requirements.txt                   # Dependency requirements
├── DINOv2_Distillation_Colab.ipynb    # Google Colab ready notebook
├── src/
│   ├── dataset.py                     # Data loaders (CIFAR-100/10) with DINOv2 transforms
│   ├── models.py                      # Teacher (DINOv2), Student (MobileNetV3), Projection Head
│   ├── losses.py                      # Composite loss (MSE feature alignment + Cross-Entropy)
│   ├── utils.py                       # Latency benchmark, t-SNE plotter, parameter counters
│   ├── train_baseline.py              # Supervised baseline training script
│   ├── train_distill.py               # Feature distillation script
│   └── evaluate.py                    # Evaluation, t-SNE extraction & ONNX export script
└── deliverables/
    ├── MidSem_Report_Outline.md       # Prepared outline & viva QA for Mid-Sem review
    └── Final_Project_Report_Template.md # Full thesis outline
```

---

## 🗓 Academic Timeline (Before Mid-Sem vs After Mid-Sem)

### Phase 1: Before Mid-Sem (Foundation & Baselines)
1. **Literature Survey & Proposal**: DINOv2 non-parametric self-distillation vs Representation Knowledge Distillation.
2. **Environment & Dataset**: Setup PyTorch data pipeline with ImageNet normalization and $224 \times 224$ resizing.
3. **Teacher Model Sandbox**: Load frozen DINOv2 ViT-S/14 via PyTorch Hub (`facebookresearch/dinov2`).
4. **Baseline Model**: Train standard supervised MobileNetV3-Small with Cross-Entropy Loss.
5. **Initial Distillation**: Build 2-layer MLP projection head and composite loss function. Run 5-epoch sanity check.
6. **Mid-Sem Viva**: Mid-sem report presentation and live feature alignment proof-of-concept.

### Phase 2: After Mid-Sem (Optimization & Benchmarks)
1. **Extended Training**: 15–30 epochs with AdamW optimizer and Cosine Annealing learning rate schedule.
2. **Hyperparameter Sweeps**: Evaluate loss balancing factor $\alpha \in \{0.1, 0.5, 1.0\}$.
3. **Edge Benchmarking**: Measure CPU & GPU latency (ms/img), FLOPs, and memory usage.
4. **t-SNE Visualizations**: Extract 384-dim latent features to demonstrate feature cluster alignment.
5. **Deployment & Thesis**: Export model to ONNX/TorchScript, write final project report, and deliver Viva Voce.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Supervised Baseline (Benchmark 1)
```bash
python src/train_baseline.py --dataset cifar100 --epochs 15 --save_path baseline_mobilenet.pth
```

### 3. Train DINOv2-Distilled MobileNetV3 (Proposed Model)
```bash
python src/train_distill.py --dataset cifar100 --epochs 15 --alpha 0.5 --save_path distilled_mobilenet.pth
```

### 4. Evaluate, Benchmark Latency & Export ONNX
```bash
python src/evaluate.py --baseline_ckpt baseline_mobilenet.pth --distilled_ckpt distilled_mobilenet.pth --export_onnx
```

### 5. Run in Google Colab
Open `DINOv2_Distillation_Colab.ipynb` directly in Google Colab, select **T4 GPU** runtime, and run all cells sequentially.
