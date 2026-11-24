# 4. Results

## Files

| File | Content |
|---|---|
| `results/tables/linear_probe_summary.csv` | all linear-probe results (clean and raw): mean/SD accuracy and macro-F1, parameters, input size, speed |
| `results/tables/finetune_summary.csv`, `finetune_runs.csv` | fine-tuning, per model and per seed |
| `results/tables/clean_vs_raw.csv` | the ten models evaluated on both label lists |
| `results/tables/silhouette_clean_vs_raw.csv` | silhouette of the frozen features under clean, raw and source-dataset labels |
| `results/tables/stats.json` | CNN vs ViT test, Kendall τ, silhouette summary |
| `results/tables/selected_models.json` | models chosen for stages B and C |
| `results/jobs/<job>/result.json` | every run of every experiment, per-class recall and F1 |
| `results/jobs/<job>/confusion*.npy` | (mean) confusion matrices |
| `results/model_info/` | feature dimension, parameters, input size and GPU speed per model |
| `results/figures/` | results PDF, t-SNE maps (+ their coordinates as CSV), pipeline figure |

Job names: `lp_clean__<model>`, `lp_raw__<model>`, `ft_clean__<model>__s<seed>`.

## Linear probe — all 40 models (clean list)

| # | Model | Family | macro-F1 (mean ± SD) | Accuracy | Params (M) | Input (px) | GPU img/s |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `vit_large_patch16_224.augreg_in21k_ft_in1k` | ViT | 0.9289 ± 0.0020 | 93.80% | 303.3 | 224 | 464 |
| 2 | `tf_efficientnet_b7.ns_jft_in1k` | CNN | 0.9269 ± 0.0018 | 93.52% | 63.8 | 600 | 113 |
| 3 | `vit_base_patch16_clip_224.openai` | ViT | 0.9217 ± 0.0018 | 92.86% | 85.8 | 224 | 1261 |
| 4 | `densenet201.tv_in1k` | CNN | 0.9153 ± 0.0030 | 92.63% | 18.1 | 224 | 1312 |
| 5 | `densenet169.tv_in1k` | CNN | 0.9103 ± 0.0023 | 92.26% | 12.5 | 224 | 1646 |
| 6 | `convnext_base.fb_in22k_ft_in1k` | CNN | 0.9086 ± 0.0019 | 92.07% | 87.6 | 224 | 885 |
| 7 | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | ViT | 0.9079 ± 0.0024 | 92.10% | 86.7 | 224 | 650 |
| 8 | `beit_base_patch16_224.in22k_ft_in22k_in1k` | ViT | 0.9047 ± 0.0028 | 91.73% | 85.8 | 224 | 1186 |
| 9 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ViT | 0.9026 ± 0.0014 | 91.64% | 85.8 | 224 | 1279 |
| 10 | `beitv2_base_patch16_224.in1k_ft_in22k_in1k` | ViT | 0.9002 ± 0.0027 | 91.36% | 85.8 | 224 | 1185 |
| 11 | `densenet121.tv_in1k` | CNN | 0.8966 ± 0.0028 | 91.03% | 7.0 | 224 | 2069 |
| 12 | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | ViT | 0.8960 ± 0.0024 | 90.91% | 48.8 | 224 | 873 |
| 13 | `convnext_small.fb_in22k_ft_in1k` | CNN | 0.8925 ± 0.0022 | 90.68% | 49.5 | 224 | 1216 |
| 14 | `vit_small_patch14_dinov2.lvd142m` | ViT | 0.8917 ± 0.0029 | 90.51% | 22.1 | 518 | 401 |
| 15 | `deit_base_patch16_224.fb_in1k` | ViT | 0.8880 ± 0.0019 | 90.31% | 85.8 | 224 | 1280 |
| 16 | `convnext_tiny.fb_in22k_ft_in1k` | CNN | 0.8871 ± 0.0018 | 90.30% | 27.8 | 224 | 1919 |
| 17 | `swin_tiny_patch4_window7_224.ms_in1k` | ViT | 0.8841 ± 0.0024 | 89.74% | 27.5 | 224 | 1402 |
| 18 | `mobilenetv3_large_100.ra_in1k` | CNN | 0.8779 ± 0.0027 | 89.39% | 4.2 | 224 | 6086 |
| 19 | `regnety_032.ra_in1k` | CNN | 0.8751 ± 0.0023 | 89.10% | 17.9 | 224 | 2078 |
| 20 | `efficientnet_b0.ra_in1k` | CNN | 0.8743 ± 0.0037 | 89.19% | 4.0 | 224 | 5001 |
| 21 | `vit_small_patch16_224.augreg_in21k_ft_in1k` | ViT | 0.8734 ± 0.0021 | 88.91% | 21.7 | 224 | 3523 |
| 22 | `resnetv2_50.a1h_in1k` | CNN | 0.8722 ± 0.0032 | 89.09% | 23.5 | 224 | 2582 |
| 23 | `efficientnet_b3.ra2_in1k` | CNN | 0.8721 ± 0.0031 | 89.05% | 10.7 | 288 | 1552 |
| 24 | `resnet50.a1_in1k` | CNN | 0.8713 ± 0.0029 | 88.78% | 23.5 | 224 | 2356 |
| 25 | `resnet101.a1h_in1k` | CNN | 0.8694 ± 0.0025 | 88.55% | 42.5 | 224 | 1610 |
| 26 | `resnet152.a1h_in1k` | CNN | 0.8669 ± 0.0029 | 88.42% | 58.1 | 224 | 1146 |
| 27 | `mobilenetv3_small_100.lamb_in1k` | CNN | 0.8663 ± 0.0024 | 88.27% | 1.5 | 224 | 20487 |
| 28 | `coat_lite_mini.in1k` | ViT | 0.8515 ± 0.0032 | 86.93% | 10.5 | 224 | 2028 |
| 29 | `pvt_v2_b2.in1k` | ViT | 0.8510 ± 0.0015 | 87.07% | 24.9 | 224 | 1392 |
| 30 | `cait_s24_224.fb_dist_in1k` | ViT | 0.8505 ± 0.0020 | 87.14% | 46.5 | 224 | 678 |
| 31 | `twins_pcpvt_small.in1k` | ViT | 0.8470 ± 0.0033 | 86.65% | 23.6 | 224 | 1755 |
| 32 | `vgg16.tv_in1k` | CNN | 0.8440 ± 0.0027 | 86.29% | 134.3 | 224 | 1481 |
| 33 | `deit_small_patch16_224.fb_in1k` | ViT | 0.8372 ± 0.0033 | 85.64% | 21.7 | 224 | 3511 |
| 34 | `xception71.tf_in1k` | CNN | 0.8350 ± 0.0036 | 85.54% | 40.3 | 299 | 540 |
| 35 | `vgg19.tv_in1k` | CNN | 0.8314 ± 0.0046 | 85.30% | 139.6 | 224 | 1295 |
| 36 | `resnet34.a1_in1k` | CNN | 0.8213 ± 0.0044 | 84.09% | 21.3 | 224 | 4951 |
| 37 | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | ViT | 0.8195 ± 0.0024 | 83.07% | 5.5 | 224 | 6636 |
| 38 | `cait_xxs24_224.fb_dist_in1k` | ViT | 0.7959 ± 0.0041 | 80.49% | 11.8 | 224 | 1182 |
| 39 | `deit_tiny_patch16_224.fb_in1k` | ViT | 0.7949 ± 0.0038 | 80.39% | 5.5 | 224 | 6827 |
| 40 | `crossvit_tiny_240.in1k` | ViT | 0.7810 ± 0.0033 | 79.84% | 6.7 | 240 | 4096 |

## Fine-tuning

| Model | Family | macro-F1 (3 seeds) | Accuracy | Linear probe macro-F1 |
|---|---|---|---|---|
| `vit_large_patch16_224.augreg_in21k_ft_in1k` | ViT | **0.9717 ± 0.0010** | 97.68% | 0.9289 |
| `densenet201.tv_in1k` | CNN | **0.9673 ± 0.0007** | 97.41% | 0.9153 |
| `densenet169.tv_in1k` | CNN | **0.9653 ± 0.0003** | 97.22% | 0.9103 |
| `vit_base_patch16_clip_224.openai` | ViT | **0.9609 ± 0.0011** | 96.87% | 0.9217 |

EfficientNet-B7 was not fine-tuned because its native input (600 px) is above the 320 px limit.

## Raw vs clean labels

| Model | Family | Raw labels (132 classes) | Clean labels (75 classes) | Difference |
|---|---|---|---|---|
| `vit_large_patch16_224.augreg_in21k_ft_in1k` | ViT | 0.9029 | 0.9289 | +0.0260 |
| `tf_efficientnet_b7.ns_jft_in1k` | CNN | 0.8980 | 0.9269 | +0.0289 |
| `vit_base_patch16_clip_224.openai` | ViT | 0.8958 | 0.9217 | +0.0259 |
| `densenet201.tv_in1k` | CNN | 0.8906 | 0.9153 | +0.0247 |
| `densenet169.tv_in1k` | CNN | 0.8843 | 0.9103 | +0.0260 |
| `convnext_base.fb_in22k_ft_in1k` | CNN | 0.8788 | 0.9086 | +0.0298 |
| `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | ViT | 0.8799 | 0.9079 | +0.0281 |
| `beit_base_patch16_224.in22k_ft_in22k_in1k` | ViT | 0.8733 | 0.9047 | +0.0314 |
| `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ViT | 0.8740 | 0.9026 | +0.0286 |
| `densenet121.tv_in1k` | CNN | 0.8694 | 0.8966 | +0.0272 |

Kendall τ between the two rankings: 0.91 (p = 3.0e-05). The two lists have
different numbers of classes, so the silhouette analysis below — which uses the same images — is the direct
comparison.

## Silhouette of frozen features (same real images, cosine)

| Model | Family | Clean labels | Raw labels | Source dataset |
|---|---|---|---|---|
| `vit_large_patch16_224` | ViT | +0.075 | -0.031 | -0.087 |
| `mobilenetv3_large_100` | CNN | +0.075 | -0.006 | -0.020 |
| `vit_small_patch14_dinov2` | ViT | +0.071 | -0.062 | -0.140 |
| `efficientnet_b0` | CNN | +0.070 | -0.010 | -0.039 |
| `densenet169` | CNN | +0.067 | -0.011 | -0.040 |
| `densenet121` | CNN | +0.067 | -0.021 | -0.056 |
| `mobilenetv3_small_100` | CNN | +0.064 | -0.015 | -0.023 |
| `densenet201` | CNN | +0.064 | -0.016 | -0.056 |
| `swin_base_patch4_window7_224` | ViT | +0.063 | -0.036 | -0.070 |
| `swin_tiny_patch4_window7_224` | ViT | +0.062 | -0.027 | -0.032 |
| `cait_s24_224` | ViT | +0.062 | -0.058 | -0.054 |
| `convnext_small` | CNN | +0.062 | -0.054 | -0.074 |
| `vit_small_patch16_224` | ViT | +0.060 | -0.043 | -0.080 |
| `resnetv2_50` | CNN | +0.060 | -0.027 | -0.047 |
| `deit_base_patch16_224` | ViT | +0.059 | -0.028 | -0.031 |
| `beit_base_patch16_224` | ViT | +0.057 | -0.059 | -0.094 |
| `resnet101` | CNN | +0.057 | -0.033 | -0.052 |
| `regnety_032` | CNN | +0.055 | -0.032 | -0.059 |
| `convnext_base` | CNN | +0.054 | -0.050 | -0.097 |
| `efficientnet_b3` | CNN | +0.054 | -0.034 | -0.041 |
| `cait_xxs24_224` | ViT | +0.053 | -0.057 | -0.056 |
| `resnet152` | CNN | +0.052 | -0.034 | -0.036 |
| `resnet50` | CNN | +0.052 | -0.047 | -0.091 |
| `vit_base_patch16_clip_224` | ViT | +0.051 | +0.012 | -0.018 |
| `beitv2_base_patch16_224` | ViT | +0.050 | -0.057 | -0.093 |
| `resnet34` | CNN | +0.050 | -0.051 | -0.066 |
| `deit_small_patch16_224` | ViT | +0.050 | -0.041 | -0.035 |
| `deit_tiny_patch16_224` | ViT | +0.049 | -0.057 | -0.038 |
| `swin_small_patch4_window7_224` | ViT | +0.049 | -0.071 | -0.083 |
| `vit_tiny_patch16_224` | ViT | +0.048 | -0.084 | -0.086 |
| `twins_pcpvt_small` | ViT | +0.048 | -0.048 | -0.044 |
| `pvt_v2_b2` | ViT | +0.048 | -0.046 | -0.040 |
| `vgg16` | CNN | +0.048 | -0.039 | -0.031 |
| `coat_lite_mini` | ViT | +0.047 | -0.045 | -0.044 |
| `crossvit_tiny_240` | ViT | +0.045 | -0.053 | -0.034 |
| `vgg19` | CNN | +0.040 | -0.050 | -0.034 |
| `convnext_tiny` | CNN | +0.038 | -0.047 | -0.078 |
| `vit_base_patch16_224` | ViT | +0.032 | -0.078 | -0.096 |
| `xception71` | CNN | +0.029 | -0.051 | -0.039 |
| `tf_efficientnet_b7` | CNN | -0.003 | -0.120 | -0.180 |

Higher is better. Negative values for the source dataset mean that images do not group by the dataset they came from.
