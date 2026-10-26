"""Script to construct and execute notebooks/05_Model_Training.ipynb with full 150-epoch results.

Presents the complete 150-epoch training dynamics for both U-Net and DeepLabV3+,
loads production checkpoint weights (checkpoints/*_best.pth), plots comparative curves,
performs real validation inference, and displays the comprehensive benchmark report.
"""

from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def build_and_run_training_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.10.11",
    }

    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Phase 6 — Semantic Segmentation Model Training (Full 150-Epoch Production Run)\n\n"
        "**Remote Sensing Scene Understanding using Semantic Segmentation (LoveDA Dataset)**\n\n"
        "### Training Configuration & Highlights\n"
        "- **Hardware & Acceleration**: NVIDIA GeForce RTX 5060 Ti GPU (15.93 GB VRAM) with Automatic Mixed Precision (AMP `fp16`)\n"
        "- **Dataset Partition**: Full LoveDA dataset (2,522 Train pairs, 1,669 Validation pairs, 512×512 resolution)\n"
        "- **Models Trained**: \n"
        "  1. **U-Net**: Classic encoder-decoder with skip connections (17.26 M parameters)\n"
        "  2. **DeepLabV3+**: ResNet34 backbone + ASPP with dilation rates `[6, 12, 18]` (26.71 M parameters)\n"
        "- **Epochs**: **150 epochs per model** (300 epochs total, Early Stopping disabled)\n"
        "- **Optimization**: AdamW (`lr=1e-4`, `weight_decay=1e-4`) with CosineAnnealingLR (`T_max=150`, `eta_min=1e-6`)\n"
        "- **Loss Function**: Combined Loss (Cross-Entropy + Multiclass Dice with `ignore_index=0`)\n"
        "- **Peak Validation mIoU**: **47.32%** (U-Net, Epoch 82) vs **43.98%** (DeepLabV3+, Epoch 66)"
    ))

    # Cell 1: Environment & Imports
    cells.append(nbf.v4.new_code_cell(
        "# Cell 1: Environment Setup and Library Imports\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "WORKSPACE_ROOT = Path('.').resolve()\n"
        "if str(WORKSPACE_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(WORKSPACE_ROOT))\n"
        "\n"
        "import torch\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from src.utils.class_mapping import (\n"
        "    RAW_CLASS_IDS,\n"
        "    CLASS_NAMES,\n"
        "    EVAL_CLASS_IDS,\n"
        "    IGNORE_INDEX,\n"
        "    NUM_CLASSES,\n"
        "    mask_to_rgb,\n"
        ")\n"
        "from src.utils.visualization import denormalize_image_tensor\n"
        "from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE\n"
        "from src.dataset.dataloader import get_dataloaders\n"
        "from src.models.unet import UNet\n"
        "from src.models.deeplabv3plus import DeepLabV3Plus\n"
        "from src.models.losses import CombinedLoss\n"
        "from src.training.trainer import SemanticSegmentationTrainer\n"
        "from src.training.training_config import (\n"
        "    CHECKPOINT_DIR,\n"
        "    DEFAULT_LR,\n"
        "    DEFAULT_WEIGHT_DECAY,\n"
        "    OUTPUTS_DIR,\n"
        "    REPORTS_DIR,\n"
        "    get_safe_device,\n"
        ")\n"
        "\n"
        "device = get_safe_device()\n"
        "print(f'Execution Device: {device}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU Name:         {torch.cuda.get_device_name(0)}')\n"
        "    print(f'VRAM Total:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n"
        "print('All training modules and configurations successfully loaded.')"
    ))

    # Cell 2: DataLoader Setup
    cells.append(nbf.v4.new_code_cell(
        "# Cell 2: LoveDA DataLoaders Configuration\n"
        "BATCH_SIZE = 8\n"
        "train_loader, val_loader = get_dataloaders(\n"
        "    batch_size=BATCH_SIZE,\n"
        "    num_workers=0,\n"
        "    target_size=TARGET_IMAGE_SIZE,\n"
        ")\n"
        "\n"
        "print(f'Dataset Resolution:    {TARGET_IMAGE_SIZE}')\n"
        "print(f'Train Loader Batches:  {len(train_loader)} batches ({len(train_loader.dataset)} sample pairs)')\n"
        "print(f'Val Loader Batches:    {len(val_loader)} batches ({len(val_loader.dataset)} sample pairs)')\n"
        "print(f'Total Batches / Epoch: {len(train_loader) + len(val_loader)} batches')"
    ))

    # Cell 3: Model 1 — U-Net 150-Epoch Training Dynamic
    cells.append(nbf.v4.new_code_cell(
        "# Cell 3: Model 1 — U-Net 150-Epoch Training Dynamic\n"
        "EPOCHS = 150\n"
        "unet_model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)\n"
        "unet_param_count = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)\n"
        "print(f'U-Net Parameter Count: {unet_param_count:,} ({unet_param_count / 1e6:.2f} M)')\n"
        "\n"
        "# Load Full 150-Epoch History\n"
        "history_path = REPORTS_DIR / 'training_history.csv'\n"
        "full_history_df = pd.read_csv(history_path)\n"
        "unet_history = full_history_df[full_history_df['model'] == 'U-Net'].copy().reset_index(drop=True)\n"
        "\n"
        "best_unet_idx = unet_history['val_mIoU'].idxmax()\n"
        "best_unet_row = unet_history.iloc[best_unet_idx]\n"
        "\n"
        "print(f'=== U-Net 150-Epoch Training Summary ===')\n"
        "print(f'  Total Epochs Completed: {len(unet_history)}')\n"
        "print(f'  Best Epoch:             {int(best_unet_row[\"epoch\"])}')\n"
        "print(f'  Peak Validation mIoU:   {best_unet_row[\"val_mIoU\"]:.2f}%')\n"
        "print(f'  Peak Validation OA:     {best_unet_row[\"val_OA\"]:.2f}%')\n"
        "print(f'  Initial Train Loss:     {unet_history[\"train_loss\"].iloc[0]:.4f} -> Final: {unet_history[\"train_loss\"].iloc[-1]:.4f}')\n"
        "print(f'  Initial Val Loss:       {unet_history[\"val_loss\"].iloc[0]:.4f} -> Final: {unet_history[\"val_loss\"].iloc[-1]:.4f}')\n"
        "\n"
        "# Plot 150-Epoch U-Net Curves\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
        "axes[0].plot(unet_history['epoch'], unet_history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)\n"
        "axes[0].plot(unet_history['epoch'], unet_history['val_loss'], label='Val Loss', color='#dc2626', linestyle='--', linewidth=2)\n"
        "axes[0].set_title('U-Net: 150-Epoch Loss Convergence', fontsize=13, fontweight='bold')\n"
        "axes[0].set_xlabel('Epoch', fontsize=11)\n"
        "axes[0].set_ylabel('Combined Loss', fontsize=11)\n"
        "axes[0].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[0].legend(fontsize=10)\n"
        "\n"
        "axes[1].plot(unet_history['epoch'], unet_history['val_mIoU'], label='Val mIoU (%)', color='#059669', linewidth=2.5)\n"
        "axes[1].plot(unet_history['epoch'], unet_history['val_OA'], label='Val OA (%)', color='#7c3aed', linestyle=':', linewidth=2)\n"
        "axes[1].scatter([best_unet_row['epoch']], [best_unet_row['val_mIoU']], color='#e11d48', s=80, zorder=5, label=f'Peak mIoU ({best_unet_row[\"val_mIoU\"]:.2f}%)')\n"
        "axes[1].set_title('U-Net: 150-Epoch Validation Metric Trajectory', fontsize=13, fontweight='bold')\n"
        "axes[1].set_xlabel('Epoch', fontsize=11)\n"
        "axes[1].set_ylabel('Score (%)', fontsize=11)\n"
        "axes[1].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[1].legend(fontsize=10)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 4: Model 2 — DeepLabV3+ 150-Epoch Training Dynamic
    cells.append(nbf.v4.new_code_cell(
        "# Cell 4: Model 2 — DeepLabV3+ 150-Epoch Training Dynamic\n"
        "deeplab_model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone='resnet34', pretrained=False)\n"
        "deeplab_param_count = sum(p.numel() for p in deeplab_model.parameters() if p.requires_grad)\n"
        "print(f'DeepLabV3+ Parameter Count: {deeplab_param_count:,} ({deeplab_param_count / 1e6:.2f} M)')\n"
        "\n"
        "# Load Full 150-Epoch History\n"
        "deeplab_history = full_history_df[full_history_df['model'] == 'DeepLabV3+'].copy().reset_index(drop=True)\n"
        "\n"
        "best_deeplab_idx = deeplab_history['val_mIoU'].idxmax()\n"
        "best_deeplab_row = deeplab_history.iloc[best_deeplab_idx]\n"
        "\n"
        "print(f'=== DeepLabV3+ 150-Epoch Training Summary ===')\n"
        "print(f'  Total Epochs Completed: {len(deeplab_history)}')\n"
        "print(f'  Best Epoch:             {int(best_deeplab_row[\"epoch\"])}')\n"
        "print(f'  Peak Validation mIoU:   {best_deeplab_row[\"val_mIoU\"]:.2f}%')\n"
        "print(f'  Peak Validation OA:     {best_deeplab_row[\"val_OA\"]:.2f}%')\n"
        "print(f'  Initial Train Loss:     {deeplab_history[\"train_loss\"].iloc[0]:.4f} -> Final: {deeplab_history[\"train_loss\"].iloc[-1]:.4f}')\n"
        "print(f'  Initial Val Loss:       {deeplab_history[\"val_loss\"].iloc[0]:.4f} -> Final: {deeplab_history[\"val_loss\"].iloc[-1]:.4f}')\n"
        "\n"
        "# Plot 150-Epoch DeepLabV3+ Curves\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
        "axes[0].plot(deeplab_history['epoch'], deeplab_history['train_loss'], label='Train Loss', color='#0284c7', linewidth=2)\n"
        "axes[0].plot(deeplab_history['epoch'], deeplab_history['val_loss'], label='Val Loss', color='#ea580c', linestyle='--', linewidth=2)\n"
        "axes[0].set_title('DeepLabV3+: 150-Epoch Loss Convergence', fontsize=13, fontweight='bold')\n"
        "axes[0].set_xlabel('Epoch', fontsize=11)\n"
        "axes[0].set_ylabel('Combined Loss', fontsize=11)\n"
        "axes[0].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[0].legend(fontsize=10)\n"
        "\n"
        "axes[1].plot(deeplab_history['epoch'], deeplab_history['val_mIoU'], label='Val mIoU (%)', color='#059669', linewidth=2.5)\n"
        "axes[1].plot(deeplab_history['epoch'], deeplab_history['val_OA'], label='Val OA (%)', color='#7c3aed', linestyle=':', linewidth=2)\n"
        "axes[1].scatter([best_deeplab_row['epoch']], [best_deeplab_row['val_mIoU']], color='#e11d48', s=80, zorder=5, label=f'Peak mIoU ({best_deeplab_row[\"val_mIoU\"]:.2f}%)')\n"
        "axes[1].set_title('DeepLabV3+: 150-Epoch Validation Metric Trajectory', fontsize=13, fontweight='bold')\n"
        "axes[1].set_xlabel('Epoch', fontsize=11)\n"
        "axes[1].set_ylabel('Score (%)', fontsize=11)\n"
        "axes[1].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[1].legend(fontsize=10)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 5: Comparative Training Dynamics Across All 150 Epochs
    cells.append(nbf.v4.new_code_cell(
        "# Cell 5: Comparative Cross-Model Training Dynamics Across All 150 Epochs\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "\n"
        "# Loss comparison\n"
        "axes[0].plot(unet_history['epoch'], unet_history['train_loss'], color='#2563eb', label='U-Net Train Loss', alpha=0.85, linewidth=1.8)\n"
        "axes[0].plot(unet_history['epoch'], unet_history['val_loss'], color='#93c5fd', linestyle='--', label='U-Net Val Loss', alpha=0.85, linewidth=1.5)\n"
        "axes[0].plot(deeplab_history['epoch'], deeplab_history['train_loss'], color='#d97706', label='DeepLabV3+ Train Loss', alpha=0.85, linewidth=1.8)\n"
        "axes[0].plot(deeplab_history['epoch'], deeplab_history['val_loss'], color='#fcd34d', linestyle='--', label='DeepLabV3+ Val Loss', alpha=0.85, linewidth=1.5)\n"
        "axes[0].set_title('Loss Convergence (300 Total Epochs)', fontsize=13, fontweight='bold')\n"
        "axes[0].set_xlabel('Epoch', fontsize=11)\n"
        "axes[0].set_ylabel('Combined Loss (CE + Dice)', fontsize=11)\n"
        "axes[0].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[0].legend(fontsize=10)\n"
        "\n"
        "# mIoU comparison\n"
        "axes[1].plot(unet_history['epoch'], unet_history['val_mIoU'], color='#2563eb', label=f'U-Net Val mIoU (Peak {best_unet_row[\"val_mIoU\"]:.2f}%)', linewidth=2.2)\n"
        "axes[1].plot(deeplab_history['epoch'], deeplab_history['val_mIoU'], color='#d97706', label=f'DeepLabV3+ Val mIoU (Peak {best_deeplab_row[\"val_mIoU\"]:.2f}%)', linewidth=2.2)\n"
        "axes[1].axvline(x=best_unet_row['epoch'], color='#2563eb', linestyle=':', alpha=0.7, label=f'U-Net Peak (Ep {int(best_unet_row[\"epoch\"])})')\n"
        "axes[1].axvline(x=best_deeplab_row['epoch'], color='#d97706', linestyle=':', alpha=0.7, label=f'DeepLab Peak (Ep {int(best_deeplab_row[\"epoch\"])})')\n"
        "axes[1].set_title('Validation mIoU Trajectories (150 Epochs)', fontsize=13, fontweight='bold')\n"
        "axes[1].set_xlabel('Epoch', fontsize=11)\n"
        "axes[1].set_ylabel('Mean IoU (%)', fontsize=11)\n"
        "axes[1].grid(True, linestyle='--', alpha=0.5)\n"
        "axes[1].legend(fontsize=10)\n"
        "\n"
        "comp_plot_path = OUTPUTS_DIR / 'models_training_comparison.png'\n"
        "plt.tight_layout()\n"
        "plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Cross-model comparative curves preserved at: {comp_plot_path}')"
    ))

    # Cell 6: Checkpoint Verification and Validation Visual Predictions
    cells.append(nbf.v4.new_code_cell(
        "# Cell 6: Load 150-Epoch Best Production Weights & Run Real Validation Sample Inference\n"
        "unet_ckpt_path = CHECKPOINT_DIR / 'unet_best.pth'\n"
        "deeplab_ckpt_path = CHECKPOINT_DIR / 'deeplabv3p_best.pth'\n"
        "\n"
        "unet_ckpt = torch.load(unet_ckpt_path, map_location=device)\n"
        "deeplab_ckpt = torch.load(deeplab_ckpt_path, map_location=device)\n"
        "\n"
        "unet_model.load_state_dict(unet_ckpt['model_state_dict'])\n"
        "deeplab_model.load_state_dict(deeplab_ckpt['model_state_dict'])\n"
        "unet_model.to(device).eval()\n"
        "deeplab_model.to(device).eval()\n"
        "\n"
        "print('Loaded Production Checkpoints:')\n"
        "print(f'  U-Net Best:      Epoch {unet_ckpt[\"epoch\"]} (Val mIoU: {unet_ckpt[\"val_miou\"]:.2f}%)')\n"
        "print(f'  DeepLabV3+ Best: Epoch {deeplab_ckpt[\"epoch\"]} (Val mIoU: {deeplab_ckpt[\"val_miou\"]:.2f}%)')\n"
        "\n"
        "# Get real validation samples\n"
        "val_images, val_masks = next(iter(val_loader))\n"
        "val_images = val_images.to(device)\n"
        "\n"
        "with torch.no_grad():\n"
        "    u_preds = torch.argmax(unet_model(val_images[:2]), dim=1).cpu().numpy()\n"
        "    d_preds = torch.argmax(deeplab_model(val_images[:2]), dim=1).cpu().numpy()\n"
        "\n"
        "gt_masks = val_masks[:2].cpu().numpy()\n"
        "rgb_imgs = [denormalize_image_tensor(val_images[i]) for i in range(2)]\n"
        "\n"
        "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n"
        "for i in range(2):\n"
        "    axes[i, 0].imshow(rgb_imgs[i])\n"
        "    axes[i, 0].set_title(f'Sample {i+1}: RGB Image', fontsize=12, fontweight='bold')\n"
        "    axes[i, 0].axis('off')\n"
        "\n"
        "    axes[i, 1].imshow(mask_to_rgb(gt_masks[i]))\n"
        "    axes[i, 1].set_title(f'Sample {i+1}: Ground Truth Mask', fontsize=12, fontweight='bold')\n"
        "    axes[i, 1].axis('off')\n"
        "\n"
        "    axes[i, 2].imshow(mask_to_rgb(u_preds[i]))\n"
        "    axes[i, 2].set_title(f'Sample {i+1}: U-Net 150-Epoch Pred', fontsize=12, fontweight='bold')\n"
        "    axes[i, 2].axis('off')\n"
        "\n"
        "    axes[i, 3].imshow(mask_to_rgb(d_preds[i]))\n"
        "    axes[i, 3].set_title(f'Sample {i+1}: DeepLabV3+ 150-Epoch Pred', fontsize=12, fontweight='bold')\n"
        "    axes[i, 3].axis('off')\n"
        "\n"
        "val_vis_path = OUTPUTS_DIR / 'validation_inference_comparison.png'\n"
        "plt.tight_layout()\n"
        "plt.savefig(val_vis_path, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved prediction overlay to: {val_vis_path}')"
    ))

    # Cell 7: Compile and Display Comprehensive 150-Epoch Report
    cells.append(nbf.v4.new_code_cell(
        "# Cell 7: Official Phase 6 150-Epoch Benchmark Report\n"
        "report_txt_path = REPORTS_DIR / 'training_report.txt'\n"
        "if report_txt_path.exists():\n"
        "    with open(report_txt_path, 'r', encoding='utf-8') as f:\n"
        "        print(f.read())\n"
        "\n"
        "summary_table = pd.DataFrame([\n"
        "    {\n"
        "        'Model': 'U-Net',\n"
        "        'Parameters': f'{unet_param_count / 1e6:.2f} M',\n"
        "        'Total Epochs': len(unet_history),\n"
        "        'Best Epoch': int(best_unet_row['epoch']),\n"
        "        'Peak Val mIoU': f'{best_unet_row[\"val_mIoU\"]:.2f}%',\n"
        "        'Peak Val OA': f'{best_unet_row[\"val_OA\"]:.2f}%',\n"
        "        'Final Train Loss': f'{unet_history[\"train_loss\"].iloc[-1]:.4f}',\n"
        "        'Checkpoint': 'checkpoints/unet_best.pth',\n"
        "    },\n"
        "    {\n"
        "        'Model': 'DeepLabV3+',\n"
        "        'Parameters': f'{deeplab_param_count / 1e6:.2f} M',\n"
        "        'Total Epochs': len(deeplab_history),\n"
        "        'Best Epoch': int(best_deeplab_row['epoch']),\n"
        "        'Peak Val mIoU': f'{best_deeplab_row[\"val_mIoU\"]:.2f}%',\n"
        "        'Peak Val OA': f'{best_deeplab_row[\"val_OA\"]:.2f}%',\n"
        "        'Final Train Loss': f'{deeplab_history[\"train_loss\"].iloc[-1]:.4f}',\n"
        "        'Checkpoint': 'checkpoints/deeplabv3p_best.pth',\n"
        "    },\n"
        "])\n"
        "\n"
        "print('\\n' + '=' * 60)\n"
        "print('PHASE 6 FINAL COMPARATIVE SUMMARY TABLE (150 EPOCHS)')\n"
        "print('=' * 60)\n"
        "display(summary_table)\n"
        "print('PHASE 6 COMPLETED: Ready for Phase 7 Comprehensive Evaluation.')"
    ))

    nb.cells = cells

    nb_path = WORKSPACE_ROOT / "notebooks/05_Model_Training.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook structure to {nb_path}")

    # Execute the notebook
    print("Executing 05_Model_Training.ipynb with full 150-epoch results...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed and saved {nb_path} successfully.")


if __name__ == "__main__":
    build_and_run_training_notebook()
