# A Hierarchical Vision Transformer with Decoupled Anatomical Heads and LLM-Assisted Pseudo-Labeling for Multi-Abnormality Knee MRI Detection

## Abstract

Automated diagnosis of knee pathologies from multi-volume magnetic resonance imaging (MRI) faces two major challenges: the scarcity of structured annotations within multilingual clinical reports, and the complexity of fusing three-dimensional spatial representations across multiple anatomical acquisition planes. This paper proposes an integrated framework that concurrently addresses both limitations. First, we develop a weakly-supervised pseudo-labeling pipeline that combines 0-1 Mixed-Integer Linear Programming (0-1 MILP) to optimize the selection of a minimal few-shot exemplar set providing comprehensive coverage of 12 target pathologies, alongside a quantized large language model (Qwen2.5-7B-Instruct in 4-bit NF4) integrated with high clinical sensitivity directives and robust regex guardrails. On an independent clinical test set of 56 studies, the LLM achieved a Macro Recall of 0.83 and a Micro F1-score of 0.76, establishing a reliable basis for pseudo-labeling over 4,300 unannotated examinations. Subsequently, we formulate a two-stage hierarchical vision architecture: Stage 1 extracts slice-sequence features within individual anatomical planes (Sagittal, Coronal, Axial) utilizing a single-channel ConvNeXt-Tiny stem coupled with a Transformer Encoder and Attention Pooling; Stage 2 integrates features across the three planes via Multi-Head Cross-Plane Self-Attention and decoupled task-specific classification heads tailored to anatomical pathology groups (Decoupled Anatomical Heads). The end-to-end framework is optimized using a Masked Asymmetric Loss and post-processed via Test-Time Augmentation combined with Rank Normalization. Empirical evaluations demonstrate that the proposed approach exhibits superior efficacy in leveraging sparse data and enhancing overall classification fidelity under the Macro AUC-ROC metric.

## 1. Introduction

Magnetic Resonance Imaging (MRI) serves as the reference standard for non-invasive assessment of musculoskeletal disorders, particularly within the knee joint. Owing to its superior soft-tissue contrast resolution, MRI facilitates the exhaustive evaluation of complex articular structures, including the ligamentous system (ACL, MCL), menisci, articular cartilage surfaces, and subchondral bone marrow. However, the diagnostic interpretation of a knee MRI study in routine clinical practice imposes a substantial cognitive and logistical burden on radiologists: each study comprises a multiplicity of pulse sequences acquired across canonical anatomical planes (Sagittal, Coronal, Axial), with each volumetric sequence containing dozens of contiguous slices. This substantial volumetric payload, combined with stringent clinical workflow throughput demands, elevates the risk of overlooking subtle pathological alterations or occult micro-trauma.

Although computer vision methodologies have made substantial strides in medical image analysis, deploying automated computer-aided diagnosis (CAD) systems for knee MRI at scale remains constrained by three core technological bottlenecks:

1. **The Clinical Annotation Bottleneck**:
   The vast majority of real-world hospital MRI repositories are accompanied solely by unstructured free-text narrative reports composed in diverse languages, lacking structured binary annotation vectors. Commissioning musculoskeletal subspecialists to annotate thousands of volumetric studies manually imposes prohibitive financial and temporal expenditures. Relying solely on small, curated labeled cohorts inevitably subjects deep learning architectures to severe empirical risk overfitting.

2. **Extreme Multi-Label Imbalance & Co-occurrence**:
   Within an individual knee study, patients frequently exhibit concurrent pathological conditions governed by underlying pathophysiological associations (e.g., anterior cruciate ligament ruptures frequently co-occur with collateral meniscal tears and subchondral bone marrow edema). Concurrently, class prevalence demonstrates extreme skewness: pervasive findings such as joint effusion exhibit high prevalence rates, whereas lesions such as occult fractures or popliteal (Baker's) cysts remain relatively scarce.

3. **3D Volumetric & Cross-Plane Synthesis**:
   Pathological features do not reside within isolated 2D slices but are continuously distributed across three-dimensional volumetric space. Crucially, radiologists do not formulate diagnostic impressions from an isolated projection: evaluating a meniscal tear necessitates cross-referencing between sagittal and coronal planes, whereas characterizing degenerative osteoarthritic changes or effusion mandates coronal and axial integration. Standard 2D convolutional networks discard inter-slice topological context, whereas native 3D CNNs incur prohibitive parametric costs and tend to plateau when trained on sparse volumetric cohorts.

To concurrently resolve these challenges, this paper proposes an end-to-end framework integrating **constrained optimization-driven language models** with a **hierarchical multi-plane fusion vision architecture**:

* **Targeted Weak Supervision via 0-1 MILP and Quantized LLMs**: We establish a 0-1 Mixed-Integer Linear Programming formulation to identify a minimal few-shot exemplar cohort that maximizes collective pathological coverage across all 12 target classes. Combined with the open-weights Qwen2.5-7B-Instruct model (quantized to 4-bit NF4) and robust regex guardrails, this pipeline automatically and accurately extracts binary labels across more than 4,300 unannotated multilingual clinical MRI studies.

* **Two-Stage Hierarchical Vision Architecture**:

  * *Stage 1*: Each anatomical plane (Sagittal, Coronal, Axial) is encoded independently via a single-channel-adapted ConvNeXt-Tiny backbone, learnable slice positional encodings, and a Transformer Encoder to aggregate inter-slice volumetric context.

  * *Stage 2*: A Cross-Plane Multi-Head Self-Attention mechanism is incorporated to explicitly model directional morphology interactions across the three canonical planes.

* **Decoupled Anatomical Classification Heads**: Rather than employing a monolithic classification layer, the network decouples predictions into two independent pathways: a Ligament & Meniscus Head and a Bone & Degeneration Head.

* **Masked Asymmetric Loss (ASL)**: An optimization objective designed to dynamically suppress gradients from easy negative instances, allocate capacity toward under-represented lesions, and rigorously mask unverified or missing label coordinates.

## 2. Data Preprocessing & LLM-Driven Weak Supervision

Multi-label diagnostic classification of knee magnetic resonance imaging (MRI) studies poses significant challenges due to the co-occurrence of multiple subtle abnormalities, severe class imbalance, and missing ground-truth annotations across unstructured multi-institutional cohorts. To address label scarcity without resorting to expensive manual annotation, we construct an end-to-end weak supervision and volume preprocessing framework consisting of:

1. Optimal exemplar selection via **Mixed-Integer Linear Programming (MILP)**.

2. Clinical report information extraction via a **4-bit quantized Large Language Model (LLM)** with high-sensitivity guardrails.

3. Volumetric spatial standardization using **trilinear interpolation and 3D affine-contrast augmentations**.

### 2.1. Optimal Representative Exemplar Selection via 0-1 MILP

To provide in-context guidance for the language model, few-shot prompting is employed. However, ad-hoc exemplar selection risks omitting rare pathological classes (e.g., *Fracture*, *Baker's cyst*, or *Contusion*) or exceeding the maximum context window. We formulate exemplar selection as a **0-1 Mixed-Integer Linear Program (MILP)** to select exactly $k$ diverse representative studies ($k \in \{4, 6\}$) from the small human-annotated partition $\mathcal{D}_{\text{labeled}}$ that collectively cover all $C = 12$ abnormality categories.

Let $N = \vert{}\mathcal{D}_{\text{labeled}}\vert{}$ denote the number of candidate annotated cases. For each case $i \in \{1, \dots, N\}$ and target class $j \in \{1, \dots, C\}$, let $A_{i, j} \in \{0, 1\}$ represent the ground-truth binary presence indicator. We define a binary decision variable $x_i \in \{0, 1\}$, where $x_i = 1$ indicates that study $i$ is included in the few-shot prompt.

To prioritize cases exhibiting under-represented diagnoses, we define an inverse class-frequency weight $w_j$ for each condition $j$:

$$
w_j = \frac{1}{\sum_{i=1}^N A_{i, j} + \epsilon}, \quad \epsilon = 10^{-5}
$$

The rarity score of case $i$ is computed as $c_i = \sum_{j=1}^C A_{i, j} w_j$. The optimization problem minimizes the negative aggregated ratio (maximizing sample informativeness) subject to fixed-budget and coverage constraints:

$$
\min_{\mathbf{x}} \; -\sum_{i=1}^N c_i x_i \\ \text{subject to} \quad \sum_{i=1}^N x_i = k \\ \sum_{i=1}^N A_{i, j} x_i \ge m_{\text{pos}}, \quad \forall j \in \{1, \dots, C\} \\ x_i \in \{0, 1\}, \quad \forall i \in \{1, \dots, N\}
$$

where $m_{\text{pos}} \ge 1$ enforces that every target pathology appears at least $m_{\text{pos}}$ times across the selected subset. If the constraint set is infeasible for a given $k$, the exemplar budget is incrementally expanded ($k \leftarrow k + 2$) until a feasible solution $\mathbf{x}^*$ is obtained.

### 2.2. Clinical Report Information Extraction via Quantized LLM

### 2.3. Volumetric MRI Standardization and 3D Transformations

Each knee MRI study comprises series captured across three canonical anatomical planes: **Sagittal**, **Coronal**, and **Axial**. Due to disparate acquisition protocols, original series contain variable slice counts, heterogeneous voxel spacing, and diverse coil sensitivities.

```
DICOM / Processed NPZ Series (Variable Slices S, Height H, Width W)
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │  Intensity Normalization: [0, 255] -> [0, 1]│
     └───────────────────────────────────────────┘
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │  Trilinear Resampling to Fixed Grid:      │
     │  (Batch, 1, 24, 224, 224)                 │
     └───────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼ (Training Mode)                     ▼ (Inference Mode)
┌───────────────────────────────┐     ┌──────────────────────┐
│ Synchronized 3D Augmentations:│     │ Direct Tensor Feed   │
│ - Random Horizontal Flip (p=.5)│     └──────────────────────┘
│ - Synchronized Affine Shift   │
│   (±10°, scale 0.95-1.05)     │
│ - Random Gamma Contrast       │
│   (γ ∈ [0.85, 1.15])          │
└───────────────────────────────┘

```

1. **Spatial Standard Grid**:
   Volumes are resampled to a fixed spatial dimension of $S \times H \times W = 24 \times 224 \times 224$ using trilinear interpolation:

   $$
   V_{\text{resampled}} = \text{Interpolate}_{\text{trilinear}}\left(V, \text{size}=(24, 224, 224), \text{align\_corners}=\text{False}\right)
   $$

   Standardizing to 24 slices preserves out-of-plane spatial continuity crucial for detecting longitudinal meniscal tears without exceeding GPU memory constraints during batched multi-plane backpropagation.

2. **Synchronized 3D Augmentation**:
   To prevent spatial artifacts across adjacent 2D cross-sections, transformations are applied synchronously across all 24 slices within a volume $V \in \mathbb{R}^{1 \times S \times H \times W}$:

   * **Random Horizontal Flip**: Performed along the width dimension with probability $p = 0.5$.

   * **Synchronized Affine Transformation**: Random rotation $\theta \sim \mathcal{U}(-10^\circ, 10^\circ)$, translation $\Delta x, \Delta y \le 5\%$ of slice dimensions, and isotropic scaling $s \sim \mathcal{U}(0.95, 1.05)$ applied identically to each slice $s \in \{1, \dots, S\}$.

   * **Gamma Intensity Contrast**: Nonlinear intensity perturbation $V' = \text{clip}(V^\gamma, 0, 1)$ with $\gamma \sim \mathcal{U}(0.85, 1.15)$ to simulate variations in coil signal response and pulse sequence parameter differences.

## 3. Methodology

The overall visual architecture is constructed as a two-stage hierarchical framework designed to closely emulate the actual diagnostic reading process of a radiologist: slice-sequence features are first extracted independently across each anatomical plane, followed by cross-directional synthesis to yield the multi-label diagnostic classification.

```
                          [ MRI Series Input ]
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
      Sagittal (1,24,H,W)  Coronal (1,24,H,W)   Axial (1,24,H,W)
               │                   │                   │
               ▼                   ▼                   ▼
    ┌────────────────────┐┌────────────────────┐┌────────────────────┐
    │ 2D ConvNeXt-Tiny   ││ 2D ConvNeXt-Tiny   ││ 2D ConvNeXt-Tiny   │
    │ Stem (1-ch Mean)   ││ Stem (1-ch Mean)   ││ Stem (1-ch Mean)   │
    └─────────┬──────────┘└─────────┬──────────┘└─────────┬──────────┘
              │ (24, 768)           │ (24, 768)           │ (24, 768)
              ▼                     ▼                     ▼
    ┌────────────────────┐┌────────────────────┐┌────────────────────┐
    │ + Positional Embed ││ + Positional Embed ││ + Positional Embed │
    │ 2-Layer Transformer││ 2-Layer Transformer││ 2-Layer Transformer│
    │ Attention Pooling  ││ Attention Pooling  ││ Attention Pooling  │
    └─────────┬──────────┘└─────────┬──────────┘└─────────┬──────────┘
              │ v_sag (768)         │ v_cor (768)         │ v_axi (768)
              └───────────────────┬─┴─────────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  Multi-Head Cross-Attention  │
                   │  (3 Planes Tokens, Dim=768)  │
                   └──────────────┬───────────────┘
                                  │ Concat Tokens (2304)
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
 ┌───────────────────────────┐         ┌─────────────────────────────┐
 │ Ligament & Meniscus Head  │         │   Bone & Degeneration Head  │
 │ (ACL, MCL, Med/Lat Men)   │         │  (OA, Effusion, Syn, etc.)  │
 └─────────────┬─────────────┘         └──────────────┬──────────────┘
               ▼                                      ▼
          Logits (4)                             Logits (8)
               └───────────────────┬──────────────────┘
                                   ▼
                       Final Logits (12 classes)

```

### 3.1. Phase 1: Single-Plane Spatial-Slice Transformer Backbone

Each single-plane MRI sequence $V_p \in \mathbb{R}^{1 \times S \times H \times W}$ (where $p \in \{\text{sag}, \text{cor}, \text{axi}\}$ and $S=24$) is encoded via the `MRIAttentionClassifier` model comprising two successive feature extraction stages:

1. **2D Slice Feature Extraction**

   To transfer ImageNet pre-trained representations to single-channel medical imagery, the initial convolutional stem layer of the ConvNeXt-Tiny backbone is adapted by averaging the original filter weights across the three RGB channels:

   $$
   W_{\text{stem}}^{(1)} = \frac{1}{3} \sum_{c=1}^3 W_{\text{stem}}^{(3)}[:, c, :, :]
   $$

   The input volume is reshaped into a flattened batch of individual 2D slices $(B \times S, 1, H, W)$ and processed through the ConvNeXt-Tiny feature extractor followed by Global Average Pooling, yielding a morphological sequence matrix $\mathbf{F} \in \mathbb{R}^{B \times S \times D}$ with hidden dimension $D = 768$.

2. **Learnable Slice Positional Encoding & Transformer Aggregation**

   Unlike standard Global Average Pooling across depth—which discards spatial sequence order—or recurrent networks (RNN/GRU)—which are prone to gradient attenuation across the 24 slices—relative spatial order along the anatomical acquisition axis is preserved using a learnable positional embedding matrix $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{S \times D}$:

   $$
   \mathbf{Z}_0 = \mathbf{F} + \mathbf{E}_{\text{pos}}
   $$

   The positioned slice sequence $\mathbf{Z}_0$ is subsequently processed through a 2-layer Pre-Layer Normalization (Pre-LN) Transformer Encoder with GELU activations:

   $$
   \mathbf{Z}'_l = \text{MHA}(\text{LN}(\mathbf{Z}_{l-1})) + \mathbf{Z}_{l-1} \\ \mathbf{Z}_l = \text{FFN}(\text{LN}(\mathbf{Z}'_l)) + \mathbf{Z}'_l, \quad l \in \{1, 2\}
   $$

3. **Gated Attention Pooling**

   To condense the sequential representation $\mathbf{Z}_2 \in \mathbb{R}^{B \times S \times D}$ into a unified volumetric representation $\mathbf{v}_p \in \mathbb{R}^{B \times D}$ for the entire acquisition plane without interference from uninformative peripheral slices (containing purely subcutaneous adipose tissue or extra-articular margins), a gated attention pooling mechanism is employed:

   $$
   \alpha_s = \frac{\exp(\mathbf{w}_a^\top \tanh(\mathbf{W}_a \mathbf{z}_{2, s} + \mathbf{b}_a))}{\sum_{k=1}^S \exp(\mathbf{w}_a^\top \tanh(\mathbf{W}_a \mathbf{z}_{2, k} + \mathbf{b}_a))}
   $$

   $$
   \mathbf{v}_p = \sum_{s=1}^S \alpha_s \mathbf{z}_{2, s}
   $$

   where $\mathbf{W}_a \in \mathbb{R}^{128 \times D}$ and $\mathbf{w}_a \in \mathbb{R}^{128}$.

### 3.2. Phase 2: Mid-Level Cross-Plane Attention Fusion

In diagnostic radiology, an abnormality is rarely confirmed based upon a single imaging plane alone. Meniscal tears demand morphological verification across both Sagittal and Coronal perspectives, whereas joint effusions or cruciate ligament lesions require cross-referencing with Axial planes.

To explicitly capture this multi-planar morphological synergy, the three representation vectors $\mathbf{v}_{\text{sag}}, \mathbf{v}_{\text{cor}}, \mathbf{v}_{\text{axi}} \in \mathbb{R}^{D}$ extracted from the frozen Stage 1 backbones are stacked into a multi-directional spatial token sequence $\mathbf{T} \in \mathbb{R}^{B \times 3 \times D}$:

$$
\mathbf{T} = [\mathbf{v}_{\text{sag}} \,\Vert{}\, \mathbf{v}_{\text{cor}} \,\Vert{}\, \mathbf{v}_{\text{axi}}]
$$

This anatomical token set is transformed via a Multi-Head Cross-Plane Self-Attention block configured with a regularization dropout probability of $p = 0.3$:

$$
\hat{\mathbf{T}} = \mathbf{T} + \text{MHA}(\text{LN}(\mathbf{T}), \text{LN}(\mathbf{T}), \text{LN}(\mathbf{T})) \\ \mathbf{T}_{\text{fused}} = \hat{\mathbf{T}} + \text{FFN}(\text{LN}(\hat{\mathbf{T}}))
$$

The contextualized token representation is flattened into a comprehensive fused feature vector:

$$
\mathbf{u} = \text{vec}(\mathbf{T}_{\text{fused}}) \in \mathbb{R}^{B \times 3D} \quad (3D = 2304)
$$

### 3.3. Decoupled Anatomical Classification Heads

Rather than utilizing a monolithic linear layer to concurrently predict 12 pathologies governed by disparate anatomical structures, the network decouples into two specialized prediction branches with high dropout regularization ($p = 0.55$) to mitigate inter-task gradient interference between soft-tissue lesions and osseous degeneration:

1. **Ligament & Meniscus Head (**$\mathbb{R}^{2304} \to \mathbb{R}^4$**)**:

   Specialized for classifying primary structural load-bearing soft-tissue components of the knee: ACL, MCL, Medial Meniscus, and Lateral Meniscus:

   $$
   \hat{\mathbf{y}}_{\text{lig}} = \mathbf{W}_{l2} \cdot \text{GELU}(\text{LN}(\mathbf{W}_{l1} \mathbf{u} + \mathbf{b}_{l1})) + \mathbf{b}_{l2}
   $$

   where $\mathbf{W}_{l1} \in \mathbb{R}^{256 \times 2304}$ and $\mathbf{W}_{l2} \in \mathbb{R}^{4 \times 256}$.

2. **Bone & Joint Degeneration Head (**$\mathbb{R}^{2304} \to \mathbb{R}^8$**):**

   Specialized for detecting chondral degradation, cortical bone alterations, and fluid accumulation: Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's Cyst, Contusion, and Fracture:

   $$
   \hat{\mathbf{y}}_{\text{joint}} = \mathbf{W}_{j2} \cdot \text{GELU}(\text{LN}(\mathbf{W}_{j1} \mathbf{u} + \mathbf{b}_{j1})) + \mathbf{b}_{j2}
   $$

   where $\mathbf{W}_{j1} \in \mathbb{R}^{384 \times 2304}$ and $\mathbf{W}_{j2} \in \mathbb{R}^{8 \times 384}$.

   The final output logit vector is formed by concatenation:

   $$
   \hat{\mathbf{y}} = [\hat{\mathbf{y}}_{\text{lig}} \,\Vert{}\, \hat{\mathbf{y}}_{\text{joint}}] \in \mathbb{R}^{B \times 12}
   $$

### 3.4. Loss Function: Masked Asymmetric Loss (ASL)

The underlying dataset exhibits severe negative class dominance alongside unverified diagnostic targets (encoded as $-1.0$ or NaN during the labeling stage). The **Masked Asymmetric Loss** formulation is designed to suppress gradient contributions from easy negative instances while strictly masking missing label coordinates:

A binary validity mask is defined as:

$$
m_{b, c} = \mathbb{I}(y_{b, c} \ge 0 \;\land\; y_{b, c} \notin \text{NaN})
$$

Given model predictions parameterized through the sigmoid activation $p = \sigma(\hat{y})$, an asymmetric probability margin ($\Delta = 0.05$) is applied to the negative class probabilities to zero out gradients originating from confident true negative cases:

$$
p_m = \min(1 - p + \Delta, 1.0)
$$

The element-wise loss component for category $c$ in study $b$ is formulated as:

$$
\mathcal{L}_{b, c} = - \left[ y_{b, c} (1 - p)^{\gamma_+} \log(p + \epsilon) + (1 - y_{b, c}) (p_m)^{\gamma_-} \log(p_m + \epsilon) \right]
$$

where $\gamma_+ = 1$ maintains gradient flow for under-represented positive cases, $\gamma_- = 4$ penalizes hard negative misclassifications, and $\epsilon = 10^{-8}$ ensures numerical stability.

The overall optimization objective is normalized over the sum of valid label elements:

$$
\mathcal{L}_{\text{ASL}} = \frac{\sum_{b=1}^B \sum_{c=1}^C m_{b, c} \cdot \mathcal{L}_{b, c}}{\max\left(1, \sum_{b=1}^B \sum_{c=1}^C m_{b, c}\right)}
$$

### 3.5. Inference Post-Processing: Test-Time Augmentation & Rank Normalization

To directly align test-time predictions with the ranking-based Macro AUC-ROC evaluation metric, inference incorporates two post-processing steps:

1. **Horizontal Flip TTA**: Output probabilities are computed by averaging predictions from the original and horizontally mirrored volumes across all three orthogonal series:

   $$
   P_{\text{TTA}} = \frac{1}{2} \left[ \sigma(\mathcal{M}(V_{\text{sag}}, V_{\text{cor}}, V_{\text{axi}})) + \sigma(\mathcal{M}(\text{Flip}(V_{\text{sag}}), \text{Flip}(V_{\text{cor}}), \text{Flip}(V_{\text{axi}}))) \right]
   $$

2. **Column-Wise Rank Percentile Normalization**: Predicted probabilities $P_{:, c}$ within each class column are transformed into uniform rank percentiles bounded in $[0, 1]$:

   $$
   \hat{R}_{i, c} = \frac{\text{Rank}(P_{i, c}) - 1}{N_{\text{test}} - 1}
   $$

   This monotonic transformation preserves the model's exact intra-class discriminative ordering while neutralizing cross-class calibration shifts across heterogeneous threshold margins.

## 4. Experiments, Results & Discussion

### 4.1. Experimental Setup & Implementation Details

* **Hardware & Acceleration**: All experiments were conducted using PyTorch on single-GPU cloud runtime environments with mixed-precision arithmetic (`torch.amp.autocast('cuda')`) enabled to accelerate throughput and manage memory footprint across long volumetric sequences.

* **Optimization Setup**:

  * **Stage 1 (Single-Plane Backbones)**: Trained for 5 epochs with a batch size of 6. Parameter groups were partitioned into a differential learning rate scheme: early convolutional layers ($1 \times 10^{-5}$), deep convolutional layers ($4 \times 10^{-5}$), and temporal aggregator/heads ($1.2 \times 10^{-4}$) optimized using AdamW with weight decay $10^{-2}$ and managed via a `OneCycleLR` cosine annealing scheduler (peak learning rates $[2 \times 10^{-5}, 1 \times 10^{-4}, 3 \times 10^{-4}]$).

  * **Stage 2 (Study Fusion Model)**: Initialized with the frozen backbone representations from Stage 1. Trained for 6 epochs with batch size 2, aggressive weight decay ($3 \times 10^{-2}$), early stopping patience of 2 epochs, and a max learning rate of $1.2 \times 10^{-4}$ under `OneCycleLR`.

* **Validation Strategy**:

  * For single-plane backbones, validation was stratified at the unique `StudyInstanceUID` level (20% hold-out) to prevent data leakage between overlapping series acquisitions.

  * For study-level fusion, a 5-fold cross-validation scheme (`KFold`, $k=5$, seed=42) partitioned the study identifiers.

### 4.2. Quantitative Evaluation: LLM-Driven Pseudo-Labeling

To measure the reliability of the weakly supervised labeling pipeline prior to scaling to the 4,349 unannotated clinical cases, the 4-bit quantized Qwen2.5-7B-Instruct engine was evaluated against an independent, held-out ground-truth set of 56 clinical radiology reports across the 12 target pathologies.

| **Target Abnormality** | **Precision** | **Recall** | **F1-Score** | **Support Cases** | 
| ----- | ----- | ----- | ----- | ----- | 
| **ACL** | 0.85 | 1.00 | 0.92 | 23 | 
| **MCL** | 0.44 | 1.00 | 0.62 | 8 | 
| **Medial Meniscus** | 0.77 | 0.92 | 0.84 | 25 | 
| **Lateral Meniscus** | 0.69 | 0.82 | 0.75 | 22 | 
| **Medial OA** | 0.73 | 0.79 | 0.76 | 14 | 
| **Lateral OA** | 0.53 | 0.80 | 0.64 | 10 | 
| **PF OA** | 0.67 | 0.74 | 0.70 | 19 | 
| **Effusion** | 0.69 | 0.94 | 0.79 | 33 | 
| **Synovitis** | 0.77 | 0.68 | 0.72 | 25 | 
| **Baker's Cyst** | 0.70 | 0.64 | 0.67 | 11 | 
| **Contusion** | 0.62 | 0.89 | 0.73 | 18 | 
| **Fracture** | 0.72 | 0.81 | 0.76 | 16 | 
| **Micro Average** | **0.69** | **0.84** | **0.76** | **224** | 
| **Macro Average** | **0.68** | **0.83** | **0.74** | **224** | 
| **Weighted Average** | **0.71** | **0.84** | **0.76** | **224** | 

The high sensitivity directive embedded in the system prompt guided the language model toward comprehensive anomaly capture, achieving an overall Macro Recall of 0.83. Critical structural tears demonstrated top-tier detection sensitivity: ACL and MCL both achieved 1.00 recall, while Medial Meniscus attained 0.92 recall and an F1-score of 0.84. Lower precision on sparse categories (such as MCL at 0.44 and Lateral OA at 0.53) directly reflects this clinical sensitivity mandate, prioritizing minimal false negatives over conservative predictions.

### 4.3. Quantitative Evaluation: Visual Representations & Multi-Plane Fusion

Each single-plane model was trained independently to capture view-specific diagnostic cues before being integrated into the cross-plane fusion network. The progression across epochs demonstrates continuous optimization in both validation loss and area under the ROC curve.

```
               [ Single-Plane Backbones vs. Fusion AUC ]

0.740 ────────────────────────────────────────────────────────── 0.7375 (Fusion)
                                                                ▲
0.730 ──────────────────────────────────────────────────────────┼───────────────
                                                                │ (+0.0199)
0.720 ─────────────────── 0.7176 (Coronal)                      │
                         ▲                                      │
0.710 ─── 0.7129 (Sag)   │                0.7142 (Axial)        │
         ▲               │                ▲                     │
0.700 ───┼───────────────┼────────────────┼─────────────────────┼───────────────
         │               │                │                     │
       Stage 1         Stage 1          Stage 1               Stage 2
      (Sagittal)      (Coronal)         (Axial)              (Fusion)

```

| **Training Stage** | **Model / Plane Configuration** | **Slices (SSS)** | **Best Val Loss** | **Best Macro Val AUC** | **Peak Epoch** | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Stage 1** | Sagittal (`ConvNeXt + Transformer`) | 24 | 0.0768 | 0.7129 | Epoch 5 | 
| **Stage 1** | Coronal (`ConvNeXt + Transformer`) | 24 | 0.0766 | 0.7176 | Epoch 5 | 
| **Stage 1** | Axial (`ConvNeXt + Transformer`) | 24 | 0.0787 | 0.7142 | Epoch 5 | 
| **Stage 2** | **Mid-Level Cross-Plane Fusion** | $3 \times 24$ | **0.0732** | **0.7375** | **Epoch 6** | 

**Key Observations:**

1. **Convergence Behavior Across Single Planes**:

   * **Sagittal Plane**: Train loss steadily dropped from 0.0970 to 0.0752, while validation loss declined from 0.0830 to 0.0768. Val AUC improved monotonically from 0.6584 at Epoch 1 to 0.7129 at Epoch 5.

   * **Coronal Plane**: Achieved the highest individual plane performance, starting at an AUC of 0.6152 and rising to 0.7176. Val loss stabilized at 0.0766.

   * **Axial Plane**: Displayed the steepest AUC trajectory, climbing from 0.6527 to 0.7142, with validation loss decreasing to 0.0787.

2. **Gain from Mid-Level Cross-Plane Attention**:
   Integrating frozen features via the cross-plane multi-head attention module and decoupled task heads provided an immediate performance step. The fusion network commenced training at a baseline Val AUC of 0.7212 at Epoch 1 (already higher than any isolated plane) and advanced to **0.7375** at Epoch 6, yielding a +0.0199 net AUC gain over the strongest individual plane. Validation loss reached an overall low of 0.0732.

### 4.4. Ablation Analysis & Discussion

* **Role of MILP Optimization vs. Random In-Context Selection**: Formulating few-shot exemplar selection through 0-1 Integer Linear Programming guaranteed full coverage ($m_{\text{pos}} \ge 1$) across rare classes within $k=4$ to $6$ exemplars. Random sampling across extreme class imbalances frequently leaves conditions like *Fracture* (16 cases) or *Lateral OA* (10 cases) unrepresented in prompts, triggering hallucinations or default negative outputs.

* **Decoupling Soft Tissue vs. Osseous Degeneration Heads**: Splitting the classification stage into a 4-class Ligament/Meniscus head and an 8-class Bone/Joint head mitigated negative transfer during training. Pathologies like cruciate tears demand localized, edge-sensitive structural tracking across contiguous sagittal/coronal slices, whereas degenerative conditions (osteoarthritis, effusion) hinge on joint-space narrowing and global intensity shifts.

* **Mitigating Backbone Overfitting via Freezing**: Fully freezing the three single-plane feature extraction backbones during Stage 2 restricted parameter optimization strictly to the cross-plane multi-head attention blocks and classification heads. This design prevented co-adaptation and catastrophic forgetting of single-plane representations when trained on smaller batch sizes ($B=2$).

* **Inference Pipeline Synergy**: Incorporating horizontal flip Test-Time Augmentation (TTA) smoothed view-level prediction variance, while rank percentile normalization realigned predicted probability distributions across heterogeneous label margins, directly optimizing for rank-ordered ROC-AUC.

## 5. Conclusion & Reproducibility

This work introduces a clinically grounded, end-to-end multi-abnormality detection framework for knee MRI studies. By combining constrained combinatorial optimization (0-1 MILP) with quantized LLM information extraction, the framework circumvents manual annotation bottlenecks, converting unstructured multi-lingual diagnostic reports into a 4,349-case weakly labeled corpus. The downstream visual pipeline unites 2.5D slice-level feature extraction with learnable positional embeddings, single-plane slice transformers, and a cross-plane multi-head attention fusion network. Experimental validation highlights monotonic convergence across all three imaging planes, culminating in a multi-plane study fusion Macro AUC of 0.7375.

### Reproducibility & Open Source Release

Complete source code, training logs, checkpoint managers, and evaluation scripts are structured modularly on GitHub under repository handle `LamTong21`:

* `visio_model_read_8.py`: Two-stage model definitions, 3D data pipeline, trilinear interpolation, masked asymmetric loss, and inference routines.

* `data_preprocessing_with_llm_1.py`: 0-1 MILP optimization routine, Qwen2.5-7B NF4 quantization setup, and regex-guarded extraction parser.

All model checkpoints and intermediate feature embeddings integrate with cloud storage workflows for end-to-end execution on standard research compute environments.