"""Script to generate Jupyter notebooks for the changes/ folder:
1. changes/05_Balanced_Model_Training.ipynb
2. changes/06_Balanced_Model_Evaluation.ipynb
"""

from pathlib import Path
import nbformat as nbf

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHANGES_DIR.mkdir(exist_ok=True)


def create_training_notebook():
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

    # Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# Phase 6 (Balanced) — Class-Imbalance Corrected Model Training\n\n"
        "**Remote Sensing Scene Understanding using Semantic Segmentation (LoveDA Dataset)**\n\n"
        "### Key Architectural & Methodological Fixes\n"
        "- **Median Frequency Balancing**: Calculates exact inverse class frequency weights from 2,522 training scenes.\n"
        "- **Balanced Loss Function**: Incorporates class weights directly into `CrossEntropyLoss` combined with Multi-Class `DiceLoss` (`ignore_index=0`).\n"
        "- **Strict Split Isolation**: \n"
        "  - **Train split** (2,522 pairs) used solely for gradient updates.\n"
        "  - **Validation split** (1,669 pairs) used solely for model evaluation and checkpointing.\n"
        "  - **Test split** strictly excluded from training.\n"
        "- **Early Stopping**: Configured with patience = 40 epochs to prevent overfitting on validation mIoU."
    ))

    # Cell 1: Environment & Setup
    cells.append(nbf.v4.new_code_cell(
        "# Cell 1: Environment Setup and Imports\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "WORKSPACE_ROOT = Path('..').resolve() if Path('.').resolve().name == 'changes' else Path('.').resolve()\n"
        "if str(WORKSPACE_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(WORKSPACE_ROOT))\n"
        "if str(WORKSPACE_ROOT / 'changes') not in sys.path:\n"
        "    sys.path.insert(0, str(WORKSPACE_ROOT / 'changes'))\n\n"
        "import torch\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "print(f'PyTorch Version: {torch.__version__}')\n"
        "print(f'CUDA Available: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'Device: {torch.cuda.get_device_name(0)}')\n"
    ))

    # Cell 2: Class Weights Inspection
    cells.append(nbf.v4.new_markdown_cell(
        "### 1. Class Distribution & Median Frequency Weighting\n\n"
        "In the raw LoveDA dataset, classes like **Barren** (4.71%) and **Building** (7.87%) are heavily under-represented "
        "compared to **Agriculture** (22.78%) and **Background** (28.78%). "
        "Using Median Frequency Weighting balances the penalty incurred when predicting rare classes."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Cell 2: Inspect and Plot Class Weights\n"
        "from changes.class_weights import get_lova_da_class_weights, LOVEDA_CLASS_FREQUENCIES\n\n"
        "weights = get_lova_da_class_weights()\n"
        "names = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Pixel Frequency\n"
        "ax1.barh(names[1:], [LOVEDA_CLASS_FREQUENCIES[i] for i in range(1, 8)], color='#4c72b0')\n"
        "ax1.set_title('Original Class Pixel Frequency (%)', fontsize=12, fontweight='bold')\n"
        "ax1.set_xlabel('Pixel Frequency %')\n"
        "ax1.grid(axis='x', linestyle='--', alpha=0.6)\n\n"
        "# Class Weights\n"
        "weights_np = weights.numpy()\n"
        "colors = ['#55a868' if w > 1.0 else '#c44e52' for w in weights_np[1:]]\n"
        "ax2.barh(names[1:], weights_np[1:], color=colors)\n"
        "ax2.axvline(1.0, color='gray', linestyle='--', linewidth=1.2)\n"
        "ax2.set_title('Calculated Loss Weights (Median Frequency)', fontsize=12, fontweight='bold')\n"
        "ax2.set_xlabel('Weight Multiplier (Higher = Stronger Penalty for Rare Class)')\n"
        "ax2.grid(axis='x', linestyle='--', alpha=0.6)\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    # Cell 3: Training History & Curves
    cells.append(nbf.v4.new_markdown_cell(
        "### 2. Balanced Training Dynamics & Convergence History\n\n"
        "The plot below tracks training loss, validation loss, validation mIoU, and overall accuracy "
        "across all completed epochs."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Cell 3: Load and Visualize Training History\n"
        "history_path = WORKSPACE_ROOT / 'changes' / 'checkpoints' / 'unet_history.csv'\n"
        "if history_path.exists():\n"
        "    df = pd.read_csv(history_path)\n"
        "    best_row = df.loc[df['val_mIoU'].idxmax()]\n"
        "    print(f\"=== U-Net Balanced Training Summary ===\")\n"
        "    print(f\"Total Epochs Logged: {len(df)}\")\n"
        "    print(f\"Peak Val mIoU:       {best_row['val_mIoU']:.2f}% (Epoch {int(best_row['epoch'])})\")\n"
        "    print(f\"Peak Val OA:         {best_row['val_OA']:.2f}% (Epoch {int(best_row['epoch'])})\")\n"
        "    print(f\"Latest Train Loss:   {df['train_loss'].iloc[-1]:.4f}\")\n\n"
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n\n"
        "    # Loss Curves\n"
        "    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss (Balanced)', color='#1f77b4', linewidth=2)\n"
        "    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)\n"
        "    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')\n"
        "    ax1.set_xlabel('Epoch')\n"
        "    ax1.set_ylabel('Loss')\n"
        "    ax1.legend()\n"
        "    ax1.grid(True, linestyle='--', alpha=0.5)\n\n"
        "    # Metrics Curves\n"
        "    ax2.plot(df['epoch'], df['val_mIoU'], label='Val mIoU (%)', color='#2ca02c', linewidth=2)\n"
        "    ax2.plot(df['epoch'], df['val_OA'], label='Val OA (%)', color='#9467bd', linewidth=2)\n"
        "    ax2.axvline(best_row['epoch'], color='red', linestyle=':', label=f'Best mIoU: {best_row[\"val_mIoU\"]:.2f}%')\n"
        "    ax2.set_title('Validation Performance Dynamics', fontsize=13, fontweight='bold')\n"
        "    ax2.set_xlabel('Epoch')\n"
        "    ax2.set_ylabel('Percentage (%)')\n"
        "    ax2.legend()\n"
        "    ax2.grid(True, linestyle='--', alpha=0.5)\n\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('History file not found yet.')\n"
    ))

    nb.cells = cells
    output_path = CHANGES_DIR / "05_Balanced_Model_Training.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {output_path}")


def create_evaluation_notebook():
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

    cells.append(nbf.v4.new_markdown_cell(
        "# Phase 7 (Balanced) — Model Evaluation & Comparison (Val Split Only)\n\n"
        "**Remote Sensing Scene Understanding using Semantic Segmentation (LoveDA Dataset)**\n\n"
        "### Evaluation Protocol\n"
        "- **Evaluation Split**: Strictly LoveDA **Validation split** (1,669 image-mask pairs).\n"
        "- **Test Split Status**: Excluded entirely.\n"
        "- **Key Objective**: Measure improvements in minority class segmentation (Barren, Building, Water, Road) "
        "resulting from the class-balanced loss formulation."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Cell 1: Setup and Environment\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "WORKSPACE_ROOT = Path('..').resolve() if Path('.').resolve().name == 'changes' else Path('.').resolve()\n"
        "if str(WORKSPACE_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(WORKSPACE_ROOT))\n"
        "if str(WORKSPACE_ROOT / 'changes') not in sys.path:\n"
        "    sys.path.insert(0, str(WORKSPACE_ROOT / 'changes'))\n\n"
        "import torch\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Run Validation Evaluation\n\n"
        "Execute the balanced evaluation script across the validation split to compute "
        "per-class IoU, overall accuracy, macro F1-score, and generate visual comparison plots."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Cell 2: Run Evaluation Script\n"
        "import subprocess\n\n"
        "eval_script = WORKSPACE_ROOT / 'changes' / 'evaluate_on_val.py'\n"
        "print('Running Validation Evaluation on Balanced Checkpoints...')\n"
        "result = subprocess.run([sys.executable, str(eval_script), '--batch_size', '8'], capture_output=True, text=True)\n"
        "print(result.stdout)\n"
        "if result.stderr:\n"
        "    print('Errors/Warnings:', result.stderr)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Inspect Benchmark Reports & Predictions\n\n"
        "Visualize the saved validation prediction comparisons and per-class performance tables."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Cell 3: View Saved Visualizations\n"
        "from PIL import Image\n"
        "import glob\n\n"
        "output_dir = WORKSPACE_ROOT / 'changes' / 'outputs'\n"
        "image_files = sorted(list(output_dir.glob('*.png')))\n\n"
        "if image_files:\n"
        "    print(f'Found {len(image_files)} prediction visualization figures:')\n"
        "    for img_path in image_files[:3]:\n"
        "        img = Image.open(img_path)\n"
        "        plt.figure(figsize=(14, 8))\n"
        "        plt.imshow(img)\n"
        "        plt.axis('off')\n"
        "        plt.title(img_path.name, fontsize=12, fontweight='bold')\n"
        "        plt.show()\n"
        "else:\n"
        "    print('Evaluation prediction plots will appear here once evaluate_on_val.py finishes.')\n"
    ))

    nb.cells = cells
    output_path = CHANGES_DIR / "06_Balanced_Model_Evaluation.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_training_notebook()
    create_evaluation_notebook()
