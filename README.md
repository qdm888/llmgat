# LLM-DA + GAT: Multimodal Sentiment Analysis Based on Large Language Model Enhancement and Graph Attention Networks

This repository contains the training and evaluation code for the proposed framework. It includes the offline LLM-driven data augmentation (LLM-DA) procedure with CLIP-guided routing, the GAT-based dynamic fusion predictor, the joint training objective, and the ablation settings used in the paper.

## Repository organization

```text
LLM-GAT-MSA/
├── README.md
├── requirements.txt
├── train.py
├── augment.py
│
├── models/
│   └── llm_gat.py
│
├── augmentation/
│   └── llm_da.py
│
├── data/
│   └── dataset.py
│
├── utils/
│   └── metrics.py
│
└── configs/
    ├── mosi.json
    ├── mosei.json
    └── sims.json
```

## Environment

The experiments reported in the paper were conducted under the following environment:

```text
Ubuntu 20.04
Python 3.8
PyTorch 1.12.0
PyTorch Geometric 2.3.0
NVIDIA GeForce RTX 4090 (24 GB)
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The scripts load pretrained checkpoints (BERT and CLIP) through Hugging Face
`transformers`. Download `bert-base-uncased` (CMU-MOSI / CMU-MOSEI),
`bert-base-chinese` (CH-SIMS), and `openai/clip-vit-base-patch32` in advance
when running on a machine without network access.

## Input data

All scripts read MMSA-style feature pickles. Each file contains the splits
`train`, `valid`, and `test`, and each split provides:

```text
text_bert           (N, 3, L)   input_ids / attention_mask / token_type_ids
audio               (N, n_a, d_a)
vision              (N, n_v, d_v)
regression_labels   (N,)
raw_text            list of N transcripts
id                  list of N sample ids
```

The feature files can be produced with MMSA-FET. The reported experiments use
BERT (text), wav2vec 2.0 (audio), and OpenFace (vision) features. Set
`feature_dims` in the dataset config to the actual (text, audio, vision)
dimensions of the features being used.

## Offline LLM-DA augmentation

The augmentation stage is performed offline and only on the training split.
For each original training sample, the routing score is the mean CLIP
text-video compatibility over 8 uniformly sampled frames. Samples with
`phi >= 0.25` are rewritten with semantic-preserving rewriting (SPR);
lower-scoring samples use sentiment-flipping rewriting (SFR). The original
audio-visual streams and the sample label are retained for every variant.

Generation settings:

```text
model        = gpt-3.5-turbo-0125
seed         = 42
temperature  = 0.7
top_p        = 1.0
max_tokens   = 128
```

Example:

```bash
python augment.py \
    --feature_path data/MOSI/unaligned_50.pkl \
    --output_path  data/MOSI/unaligned_50_llmda.pkl \
    --frames_root  data/MOSI/frames \
    --phi_cache    data/MOSI/clip_phi.json \
    --tokenizer bert-base-uncased
```

The script performs the following procedure:

1. load the original training split and read each sample's transcript;
2. uniformly sample 8 frames per clip from `frames_root` (`<id>.npz` with a
   `frames` array, or `<id>/*.jpg`);
3. compute the CLIP routing score and cache it in `--phi_cache`;
4. call the LLM with the SPR or SFR prompt according to the routing rule;
5. tokenize each rewritten text with the dataset's BERT tokenizer;
6. write the augmented pickle with the training split doubled
   (original + variant) and the routing metadata stored under
   `augmentation`.

Use `--tokenizer bert-base-chinese` for CH-SIMS. Validation and test samples
are never added to the training set.

## Training settings

Unless otherwise specified, the experiments use:

```text
optimizer           = Adam
learning rate       = 1e-4
weight decay        = 1e-5
batch size          = 32
maximum epochs      = 100
early stopping      = patience 15
dropout             = 0.3
projection dim      = 256
graph window w      = 3
GAT layers          = 3
attention heads     = 8
consistency weight  = 0.1
random seed         = 1111
```

The joint training objective is `L = L_cls + lambda * L_cons`, where `L_cons`
is the cosine modality-consistency term computed from the post-GAT modality
vectors.

## Reproducing the main experiment

Train on the augmented feature file:

```bash
python train.py --dataset mosi \
    --feature_path data/MOSI/unaligned_50_llmda.pkl
```

The script performs the following procedure:

1. load the train/valid/test splits from the feature pickle;
2. train the GAT predictor with the joint objective;
3. select the checkpoint with the lowest validation MAE;
4. evaluate the selected checkpoint on the test split;
5. report Non-0 Acc-2, F1, Acc-7 (Acc-3 for CH-SIMS), MAE, and Corr.

Use `--dataset mosei` or `--dataset sims` with the corresponding feature file
for the other benchmarks. Per-dataset hyperparameters are in `configs/`.

## Ablation experiments

The augmentation and fusion settings are controlled independently.

### No augmentation (GAT only)

Train on the original feature file:

```bash
python train.py --dataset sims --feature_path data/SIMS/unaligned_39.pkl
```

### Concat fusion (LLM-DA only)

Replace the GAT with mean-pooling concatenation via `--fusion_mode concat`:

```bash
python train.py --dataset sims --fusion_mode concat \
    --feature_path data/SIMS/unaligned_39_llmda.pkl
```

### Single-strategy augmentation

Force one rewriting strategy for all training samples:

```bash
python augment.py --strategy spr ...   # SPR only
python augment.py --strategy sfr ...   # SFR only
```

The CLIP routing score is still recorded in the output metadata.

### Hyperparameter sensitivity

Edit `graph_window` (w) or `consistency_weight` (lambda) in the dataset config
and evaluate on the validation split to reproduce the sensitivity analysis.

## Output

`train.py` prints the metric set for each split and saves the best checkpoint
under `saved_models/`. Passing `return_attn=True` to the model additionally
returns the head-averaged edge attention of the last GAT layer
`(N, 3L, 3L)`, which is used for the attention visualization in the paper.
