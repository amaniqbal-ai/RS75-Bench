# 2. Dataset

## Sources

The pool contains these 15 public scene-classification datasets (folder names as used in our manifests):

```
01_RSI_CB128   03_EuroSAT    05_PatternNet   06_AID          07_NWPU_RESISC45
08_RSSCN7      09_RSI-CB256  11_SIRI_WHU     CLRS            OPTIMAL-31
RSD46-WHU      RSICD_images_correct          RSITMD          UCMerced_LandUse   WHU-RS19
```

Download them from their authors and place them in one folder called `datasets_initail/` (spelling as used by the
scripts), keeping each dataset's own class folders.

## The two label lists

| List | Images | Classes | Built by |
|---|---|---|---|
| Raw | 216,472 | 132 | original folder names, synonyms merged only by spelling |
| Clean | 200,183 real (+41,252 augmented) | 75 | every image checked by eye; class decisions made independently by three people, who agreed in every case |

Files:

| File | Content |
|---|---|
| `data/manifests/clean_manifest.csv.gz` | clean class, file name and source path of every clean image |
| `data/manifests/raw_manifest.csv.gz` | raw class, file name and source path of every raw image |
| `data/lists/index_*.csv.gz` | the exact image lists used in the experiments (`clean`, `cleanlp` = real images only, `raw`) |
| `data/splits/split_*_s<seed>.npz` | train / val / test indices into those lists, seeds 0–9 |
| `data/classes_clean_75.txt`, `data/classes_raw_132.txt` | class names |
| `data/clean_class_counts.csv` | real and augmented images per clean class |
| `data/dataset_stats.json` | counts used in the paper |

## Splits and leakage

* Only real images are split: 70 / 15 / 15 per class, ten fixed seeds.
* Augmented copies (file names ending in `_augNNNN` or `_topupNNNN`) are used **only for fine-tuning** and only when
  their source image is in the training set. Validation and test sets contain real images only.
* The linear probe never sees augmented images.

## Rebuilding the datasets

```bash
cd dataset_tools
python build_datasets.py --orig ../datasets_initail --package ../upload_package --out ../rebuilt
```

`upload_package/` holds the two manifests (from `data/manifests/`, unzipped) and, for the clean list, the folder
`extras/` with the augmented copies (available on request). Without `extras/`, all real images — everything the
linear-probe experiments need — are rebuilt from the public datasets.

## How the lists were made (dataset_tools/)

| Script | Purpose |
|---|---|
| `find_duplicates.py` | exact (SHA-256) and near (perceptual hash) duplicates across the 15 datasets, dataset × dataset tables, label conflicts |
| `match_to_originals.py` | matches each image of the raw-label copy to its original file |
| `finalize_raw_labels.py` | accepts the matches (normal, identical duplicates, clear matches) and writes the raw list |
| `build_raw_folders.py` | writes the raw list as 132 image folders |
| `make_package.py` | writes the manifests and collects images that do not exist in the originals |
| `build_datasets.py` | rebuilds both lists anywhere from the originals and the package |

Main audit findings: OPTIMAL-31 is contained in NWPU-RESISC45 (1,860 identical files), RSICD contains resized copies of
AID, most of RSITMD comes from RSICD, AID and WHU-RS19 share several hundred images, and 335 duplicate pairs carry
different class names in different datasets.
