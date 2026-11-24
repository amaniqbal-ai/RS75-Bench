"""Paths, settings and the list of models used in the benchmark."""

import re
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent  # folder that contains main.py
DATA = ROOT / "data"
RES = ROOT / "results"
FEAT = ROOT / "features"
LOGS = ROOT / "logs"
STATE = ROOT / "state"
RUNTIME = ROOT / "runtime"
for _folder in (DATA, RES, FEAT, LOGS / "jobs", STATE, RUNTIME):
    _folder.mkdir(parents=True, exist_ok=True)

STOP_FLAG = STATE / "STOP"
AUG_RE = re.compile(r"_(?:aug|topup)\d{4}$")  # file-name suffix of augmented copies
DATA_VERSION = 2

# ---------------------------------------------------------------- models (timm names)
CNN = [
    "tf_efficientnet_b7.ns_jft_in1k",
    "resnetv2_50.a1h_in1k",
    "resnet101.a1h_in1k",
    "convnext_base.fb_in22k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "densenet121.tv_in1k",
    "efficientnet_b3.ra2_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "resnet152.a1h_in1k",
    "resnet50.a1_in1k",
    "densenet201.tv_in1k",
    "xception71.tf_in1k",
    "vgg19.tv_in1k",
    "densenet169.tv_in1k",
    "efficientnet_b0.ra_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "regnety_032.ra_in1k",
    "vgg16.tv_in1k",
    "resnet34.a1_in1k",
    "mobilenetv3_small_100.lamb_in1k",
]
VIT = [
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit_base_patch16_224.fb_in1k",
    "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    "beit_base_patch16_224.in22k_ft_in22k_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "twins_pcpvt_small.in1k",
    "coat_lite_mini.in1k",
    "cait_xxs24_224.fb_dist_in1k",
    "vit_base_patch16_clip_224.openai",
    "cait_s24_224.fb_dist_in1k",
    "pvt_v2_b2.in1k",
    "crossvit_tiny_240.in1k",
    "deit_tiny_patch16_224.fb_in1k",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "deit_small_patch16_224.fb_in1k",
]
# run the slowest models first, so memory problems show up early
HEAVY_FIRST = [
    "tf_efficientnet_b7.ns_jft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch14_dinov2.lvd142m",
    "xception71.tf_in1k",
    "vgg19.tv_in1k",
    "vgg16.tv_in1k",
]
# tiny models for the CPU test mode
TEST_MODELS = {
    "CNN": ["mobilenetv3_small_100.lamb_in1k", "resnet34.a1_in1k"],
    "ViT": ["deit_tiny_patch16_224.fb_in1k", "vit_tiny_patch16_224.augreg_in21k_ft_in1k"],
}

# ---------------------------------------------------------------- settings
# every value can be overridden in settings.ini (key=value)
DEFAULTS = {
    "hf_repo": "aman2625/rs-data",
    "run_name": "",
    # linear probe
    "lp_runs": "10",
    "svm_epochs": "200",
    "svm_lr": "5e-3",
    "svm_batch": "2048",
    "alphas": "1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10",
    "feature_batch": "256",
    # model selection for the raw labels and for fine-tuning
    "raw_top_per_family": "5",
    "ft_top_per_family": "2",
    "ft_max_input": "320",
    # fine-tuning
    "ft_seeds": "3",
    "ft_epochs": "10",
    "ft_batch": "64",
    "ft_lr": "1e-4",
    # augmentation top-up and embedding analysis
    "topup_target": "2000",
    "topup_threshold": "1400",
    "tsne_per_class": "40",
    # hardware
    "workers_per_job": "auto",
    "gpus": "auto",
    # uploads and messages
    "upload_every_min": "10",
    "progress_every_min": "60",
    "drive_remote": "gdrive",
    "drive_folder": "Satellite",
    # CPU test mode (tiny data, tiny models)
    "test_mode": "0",
    "test_images_per_class": "12",
}


def read_kv(path):
    """Read a simple key=value file (lines starting with # are comments)."""
    values = {}
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_settings():
    settings = dict(DEFAULTS)
    settings.update(read_kv(ROOT / "settings.ini"))
    return settings


CFG = load_settings()
TEST_MODE = CFG["test_mode"] == "1"


def short_name(model):
    """'resnet50.a1_in1k' -> 'resnet50'"""
    return model.split(".")[0]


def models_by_family():
    return TEST_MODELS if TEST_MODE else {"CNN": CNN, "ViT": VIT}


def family_of(model):
    for family, models in models_by_family().items():
        if model in models:
            return family
    return "?"
