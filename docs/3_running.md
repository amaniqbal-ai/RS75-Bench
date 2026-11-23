# 3. Running the benchmark

## What it does

| Stage | Content |
|---|---|
| 0 data | download (or use local zips), unzip, image lists, fixed splits, augmentation top-up |
| 1 smoke test | checks the GPUs, loads every model, runs a mini linear probe and a mini fine-tune |
| A | linear probe of all 40 models on the clean list (real images, 10 splits) |
| B | linear probe of the best 5 CNNs and best 5 ViTs on the raw list |
| C | fine-tuning of the best 2 CNNs and best 2 ViTs (3 seeds, 10 epochs) |
| R | tables, statistics, t-SNE / silhouette analysis and a results PDF |

Finished work is saved after every job (and after every fine-tuning epoch) and is never repeated.

## Windows (one click)

1. Copy the `benchmark` folder to a short path on a fast disk, e.g. `C:\SatBench`.
2. Put `hf_token.txt` (one line `token=hf_...`) next to `RUN.bat` — or put `datasets_initail.zip` and
   `upload_package.zip` there yourself.
3. Optional: `notify.txt` (phone messages) and `rclone.conf` (Google Drive upload).
4. Double-click `RUN.bat`. Stop with `STOP.bat`; continue by double-clicking `RUN.bat` again.
5. When it has finished, `COLLECT.bat` packs the results into `COLLECT_OUTPUT/`.

`README_WINDOWS.txt` is a one-page version of this guide for the person operating the machine.

## Command line

```bash
cd benchmark
python main.py run           # run or continue
python main.py job smoke     # a single job
python main.py report        # rebuild the report from finished results
```

All GPUs reported by `nvidia-smi` are used, one job per GPU. To use fewer, set `gpus=1` (or 2, 3) in `settings.ini`.

## Settings (`benchmark/settings.ini`)

| Key | Default | Meaning |
|---|---|---|
| `lp_runs` | 10 | linear-probe repetitions (different splits) |
| `raw_top_per_family` | 5 | models per family re-tested on the raw list |
| `ft_top_per_family` | 2 | models per family that are fine-tuned |
| `ft_seeds`, `ft_epochs`, `ft_batch`, `ft_lr` | 3, 10, 64, 1e-4 | fine-tuning |
| `ft_max_input` | 320 | models with a larger native input are not fine-tuned |
| `topup_target`, `topup_threshold` | 2000, 1400 | augmentation top-up for small classes |
| `tsne_per_class` | 40 | real images per class for t-SNE / silhouette |
| `gpus`, `workers_per_job` | auto | hardware |
| `test_mode` | 0 | 1 = tiny CPU test run |

## Optional services

**Phone messages (ntfy).** Install the ntfy app, subscribe to a private topic and write it in `notify.txt`
(`ntfy_topic=...`). You get: started, smoke test result, hourly progress, errors and the final PDF.

**Google Drive (rclone).** Create a remote called `gdrive` with `rclone config` and copy `rclone.conf` next to
`RUN.bat`. Results and logs are uploaded every 10 minutes (model checkpoints and features are not).

Keep `hf_token.txt`, `rclone.conf` and `notify.txt` private; they are listed in `.gitignore` and are never uploaded or
packed by `COLLECT.bat`.

## Run time (one RTX A6000)

| Stage | Time |
|---|---|
| A: 40 models | 5.7 h |
| B: 10 models | 2.2 h |
| C: 12 fine-tuning runs | 23.6 h (ViT-L ≈ 3.8 h per run) |

Measured on the authors' run (one A6000, data on an HDD).

With four GPUs the jobs run in parallel and the total drops to roughly a third.
