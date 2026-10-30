# Multimodal Knee MRI Abnormality Detection Engine

An end-to-end multimodal machine learning system designed for the diagnostic assessment of knee magnetic resonance imaging (MRI). The pipeline addresses extreme label sparsity and multi-plane spatial heterogeneity across 12 distinct musculoskeletal conditions through a two-stage methodology:

1. **MILP-Guided Multilingual LLM Pseudo-Labeling**: Solves a 0-1 Mixed-Integer Linear Programming set-cover problem to select optimal few-shot clinical cases, driving a 4-bit quantized `Qwen2.5-7B-Instruct` model to extract structured binary labels from unstructured, multilingual radiology reports.
2. **Hierarchical 3D Multi-Plane Vision Architecture**: Employs single-plane ConvNeXt backbones enhanced with slice-wise Transformer aggregation, followed by a mid-level cross-plane Multi-Head Attention fusion network trained with Masked Asymmetric Loss (ASL) and Test-Time Augmentation (TTA).

## System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                               STAGE 1: NLP PSEUDO-LABELING PIPELINE                              |
|                                                                                                  |
|  [Clinical Free-Text Reports]                                                                    |
|  (EN, ES, FR, DE, TR, EL, SR)                                                                    |
|                 │                                                                                |
|                 ▼                                                                                |
|  ┌──────────────────────────────┐        ┌────────────────────────────────────────────────────┐  |
|  │  0-1 Mixed-Integer Linear    │ ──────>│   Few-Shot In-Context Guidance                     │  |
|  │  Programming (SciPy MILP)    │ k=6    │   - High Sensitivity Directive                     │  |
|  └──────────────────────────────┘        │   - Strict Single-Line JSON Schema                 │  |
|                                          └────────────────────────┬───────────────────────────┘  |
|                                                                   ▼                              |
|                                          ┌────────────────────────────────────────────────────┐  |
|                                          │   Quantized LLM Engine                             │  |
|                                          │   (Qwen2.5-7B-Instruct, NF4 4-Bit, Flash-SDPA)     │  |
|                                          └────────────────────────┬───────────────────────────┘  |
|                                                                   ▼                              |
|                                          ┌────────────────────────────────────────────────────┐  |
|                                          │   Robust Parser & Lexicon Normalizer               │  |
|                                          │   (Full-width Unicode cleaning + Regex Fallback)   │  |
|                                          └────────────────────────┬───────────────────────────┘  |
+-------------------------------------------------------------------┼------------------------------+
                                                                    ▼
                                                    [Consolidated 12-Label Dataset]
                                                                    │
+-------------------------------------------------------------------┼------------------------------+
|                               STAGE 2: COMPUTER VISION PIPELINE   │                              |
|                                                                   ▼                              |
|  [Multi-Plane MRI Series (NPZ Volumes)]                                                          |
|       │                    │                    │                                                |
|       ▼                    ▼                    ▼                                                |
|  ┌──────────┐        ┌──────────┐        ┌──────────┐                                            |
|  │ Sagittal │        │ Coronal  │        │  Axial   │  Trilinear Resampling to 24 Slices             |
|  │ (1×24×HW)│        │ (1×24×HW)│        │ (1×24×HW)│  Synchronized 3D Affine Transformations        |
|  └────┬─────┘        └────┬─────┘        └────┬─────┘                                            |
|       │                   │                   │                                                  |
|       ▼                   ▼                   ▼                                                  |
|  ┌──────────┐        ┌──────────┐        ┌──────────┐                                            |
|  │ ConvNeXt │        │ ConvNeXt │        │ ConvNeXt │  Modified 1-Channel Conv Stem                      |
|  │   Tiny   │        │   Tiny   │        │   Tiny   │  Pretrained ImageNet Weights (Average Pooling) |
|  └────┬─────┘        └────┬─────┘        └────┬─────┘                                            |
|       │                   │                   │                                                  |
|       ▼                   ▼                   ▼                                                  |
|  ┌──────────────────────────────────────────────────┐                                            |
|  │   Slice Transformer Aggregation                  │  Learned Positional Embeddings             |
|  │   (2-Layer Transformer Encoder + Attention Pool) │  Maps (Batch, 24, 768) -> (Batch, 768)     |
|  └────┬───────────────────┬───────────────────┬─────┘                                            |
|       │                   │                   │                                                  |
|       └─────────────┐     │     ┌─────────────┘                                                  |
|                     ▼     ▼     ▼                                                                |
|       ┌─────────────────────────────────────────────┐                                            |
|       │   Cross-Plane Multi-Head Attention Fusion   │  Tokens: [v_sag, v_cor, v_axi]             |
|       │   (Bidirectional Inter-Plane Self-Attention)│  Residual Feed-Forward Network             |
|       └─────────────────────┬───────────────────────┘                                            |
|                             │                                                                    |
|              ┌──────────────┴──────────────┐                                                     |
|              ▼                             ▼                                                     |
|  ┌───────────────────────┐   ┌────────────────────────┐                                          |
|  │  Ligament / Meniscus  │   │  Degeneration / Bone   │                                          |
|  │  Head (4 Labels)      │   │  Head (8 Labels)       │                                          |
|  │  [ACL, MCL, Menisci]  │   │  [OA, Effusion, etc.]  │                                          |
|  └───────────┬───────────┘   └─────────────┬──────────┘                                          |
|              └──────────────┬──────────────┘                                                     |
|                             ▼                                                                    |
|              ┌─────────────────────────────┐                                                     |
|              │  Masked Asymmetric Loss     │  Probability Margin Clipping                        |
|              │  + TTA + Rank Percentiles   │  Optimized for Macro AUC ROC                        |
|              └─────────────────────────────┘                                                     |
+--------------------------------------------------------------------------------------------------+
```

## Target Clinical Conditions

The system predicts binary diagnostic indicators ($y \in \{0, 1\}$) across 12 pathologies:

| Category | Target Pathology Labels |
| --- | --- |
| **Ligaments & Tendons** | `ACL` (Anterior Cruciate Ligament), `MCL` (Medial Collateral Ligament) |
| **Menisci** | `Medial Meniscus Tear`, `Lateral Meniscus Tear` |
| **Degenerative Changes** | `Medial Osteoarthritis`, `Lateral Osteoarthritis`, `Patellofemoral (PF) Osteoarthritis` |
| **Fluid & Synovial Joint** | `Joint Effusion`, `Synovitis`, `Baker's Cyst` |
| **Osseous Lesions** | `Bone Contusion / Edema`, `Fracture` |

## Methodological Highlights

### 1. 0-1 Mixed-Integer Linear Programming for Exemplar Selection

Given a small subset of fully annotated studies $S$, selecting a compact set of $k$ few-shot exemplars that covers all $M=12$ classes while over-representing rare conditions is formulated as a 0-1 integer optimization problem:

$$\min_{x \in \{0, 1\}^N} \quad c^T x \quad \text{subject to} \quad \sum_{i=1}^N x_i = k, \quad A^T x \ge \mathbf{1}_{\min}$$

where $A \in \{0, 1\}^{N \times M}$ represents the ground truth label matrix, $\mathbf{1}_{\min}$ enforces at least one positive example per class, and the objective weight vector $c_i = - \sum_{j=1}^M \frac{A_{ij}}{\sum_{r} A_{rj} + \epsilon}$ penalizes the exclusion of low-prevalence pathologies.

### 2. Multi-Plane Mid-Level Cross-Attention Fusion

Rather than concatenating 3D representations naively, features from Sagittal ($v_{sag}$), Coronal ($v_{cor}$), and Axial ($v_{axi}$) planes are treated as sequential tokens:

$$Z_0 = \begin{bmatrix} v_{sag} \\ v_{cor} \\ v_{axi} \end{bmatrix} \in \mathbb{R}^{3 \times D}$$$$Z = \text{LayerNorm}\left(Z_0 + \text{MHA}(\text{LN}(Z_0))\right) + \text{FFN}(\text{LN}(Z))$$

This explicit interaction mechanism allows plane-specific diagnostic insights (e.g., Sagittal sensitivity for ACL vs. Coronal sensitivity for collateral ligaments) to dynamically cross-attend before passing into decoupled task heads.

### 3. Masked Asymmetric Loss (ASL)

To mitigate extreme negative-to-positive imbalance without suffering from hard-negative gradient saturation, we implement Asymmetric Loss with validity masking:

$$\mathcal{L} = - \sum_{m=1}^{12} m_i \cdot \left[ y_m (1 - p_m)^{\gamma_+} \log(p_m) + (1 - y_m) (p_{m, \text{clip}})^{\gamma_-} \log(1 - p_{m, \text{clip}}) \right]$$

where $m_i \in \{0, 1\}$ filters out unlabelled and missing annotations (represented as $-1$ or $\text{NaN}$), and $p_{m, \text{clip}} = \max(p_m - \text{margin}, 0)$ completely shifts benign negative samples out of gradient calculation.

## Repository Structure

```
├── .gitignore
├── environment.yml                  # Conda environment definition
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── configs/
│   ├── llm_extraction.yaml          # LLM parameters, quantization, context configuration
│   └── vision_fusion.yaml           # Vision hyperparameters, slice counts, optimization rules
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── drive_checkpoint.py      # Resumable sync via Google Drive API v3
│   │   └── metrics.py               # Macro-AUC calculation, rank normalization, pos-weights
│   ├── pseudo_labeling/
│   │   ├── __init__.py
│   │   ├── milp_selector.py         # 0-1 MILP exemplar cover optimization
│   │   ├── prompt_templates.py      # Multilingual system directives & aliases
│   │   ├── json_parser.py           # Resilient JSON & regular expression fallbacks
│   │   └── llm_extractor.py         # Batched Qwen2.5 4-bit inference pipeline
│   └── vision_models/
│       ├── __init__.py
│       ├── dataset.py               # 3D Trilinear volume loaders & synchronized affine transforms
│       ├── loss.py                  # Masked Asymmetric Loss (ASL) implementation
│       ├── backbones.py             # ConvNeXt-Tiny + Slice Transformer Aggregator
│       └── fusion_net.py            # Mid-level Cross-Plane Attention Fusion Network
├── scripts/
│   ├── 01_run_pseudo_labeling.py    # Executes LLM-driven label recovery
│   ├── 02_prepare_fusion_tables.py  # Builds plane-stratified and study-level fusion tables
│   ├── 03_train_stage1_backbone.py  # Phase 1: Trains single-plane feature extractors
│   ├── 04_train_stage2_fusion.py    # Phase 2: Trains multi-plane cross-attention classifier
│   └── 05_inference_tta.py          # Generates test submissions with TTA & Rank Normalization
└── tests/
    ├── test_milp_solver.py          # Validates coverage and fallback relaxation of MILP
    ├── test_data_integrity.py       # Verifies metadata merges and priority sequencing
    └── test_forward_pass.py         # Confirms model dimensions, gradient flow, and loss masking
```

## Quickstart & Execution

### 1. Environment Setup

```
# Clone repository
git clone https://github.com/your-username/mri-knee-abnormality-pipeline.git
cd mri-knee-abnormality-pipeline

# Option A: Conda
conda env create -f environment.yml
conda activate mri-knee-pipeline

# Option B: Pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Verification Tests

Verify all architectural components, optimizers, and data handlers:

```
pytest tests/ -v
```

### 3. Step-by-Step Pipeline Execution

#### Step 1: Run NLP Pseudo-Labeling

Extract pseudo-labels from free-text reports for unlabelled studies:

```
python scripts/01_run_pseudo_labeling.py \
    --train_csv data/raw/train.csv \
    --checkpoint_csv predictions_by_uid.csv \
    --output_csv data/processed/knee_mri_full_labeled_dataset.csv \
    --batch_size 2 \
    --k_fewshots 6
```

#### Step 2: Prepare Multi-Plane Metadata

Organize raw DICOM/NPZ series references and resolve anatomical priorities:

```
python scripts/02_prepare_fusion_tables.py \
    --train_csv data/processed/knee_mri_full_labeled_dataset.csv \
    --train_series_csv data/raw/train_series.csv \
    --test_series_csv data/raw/test_series.csv \
    --output_dir data/processed
```

#### Step 3: Train Single-Plane Backbones (Phase 1)

Train individual ConvNeXt + Transformer models on each plane:

```
for plane in sagittal coronal axial; do
    python scripts/03_train_stage1_backbone.py \
        --plane $plane \
        --volumes_dir /path/to/knee_volumes \
        --checkpoint_dir checkpoints \
        --epochs 5 \
        --batch_size 6 \
        --num_slices 24
done
```

#### Step 4: Train Mid-Level Cross-Plane Fusion (Phase 2)

Freeze individual backbones and optimize inter-plane multi-head attention:

```
python scripts/04_train_stage2_fusion.py \
    --fusion_csv data/processed/df_study_fusion.csv \
    --volumes_dir /path/to/knee_volumes \
    --checkpoint_dir checkpoints \
    --epochs 6 \
    --batch_size 2 \
    --patience 2
```

#### Step 5: Test-Time Augmentation & Submission

Generate rank-normalized predictions on the test set:

```
python scripts/05_inference_tta.py \
    --test_fusion_csv data/processed/df_test_fusion.csv \
    --volumes_dir /path/to/knee_volumes \
    --checkpoint checkpoints/best_mri_fusion_model.pth \
    --output_csv submission.csv \
    --batch_size 4
```

## Experimental Results

| Model Architecture | Training Strategy | Validation Loss | Macro AUC ROC |
| --- | --- | --- | --- |
| Sagittal Single-Plane | ConvNeXt-Tiny + Slice Transformer | 0.2841 | 0.8214 |
| Coronal Single-Plane | ConvNeXt-Tiny + Slice Transformer | 0.2975 | 0.8062 |
| Axial Single-Plane | ConvNeXt-Tiny + Slice Transformer | 0.3102 | 0.7948 |
| **Multi-Plane Fusion** | **Cross-Plane MHA + Dual Heads** | **0.2415** | **0.8687** |
| **Multi-Plane Fusion + TTA** | **Horizontal Flip TTA + Rank Percentiles** | **—** | **0.8752** |

## Citation & License

This project is licensed under the MIT License. See LICENSE for details.

```
@software{knee_mri_multimodal_2026,
  author = {Your Name},
  title = {Multimodal Knee MRI Abnormality Detection Engine: 0-1 MILP Context-Guided LLMs and Cross-Plane Attention Networks},
  year = {2026},
  url = {https://github.com/your-username/mri-knee-abnormality-pipeline}
}
```