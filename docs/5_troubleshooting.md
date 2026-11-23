# 5. Troubleshooting

| Message | Cause and fix |
|---|---|
| `SETUP FAILED` | usually the internet connection; run `RUN.bat` again. Still failing: delete `runtime/` and run again. Details in `logs/setup.log`. |
| `No NVIDIA GPU found` | install or update the NVIDIA driver (`nvidia-smi` must work). |
| `PyTorch cannot use the GPUs` | update the driver, delete `runtime/`, run again. |
| `Smoke test FAILED` | see `logs/jobs/smoke.log`; nothing else was started. |
| `Hugging Face refused access` | check `hf_token.txt` (`token=hf_...`) or place the two zips next to `RUN.bat`. |
| `No rclone.conf next to RUN.bat` | the file must sit in the same folder as `RUN.bat` and be named exactly `rclone.conf` (Windows may hide a `.txt` extension). No restart needed. |
| `401` / `invalid_grant` (Drive) | the Google login expired: run `rclone config reconnect gdrive:` and replace `rclone.conf`. |
| `GPU problem (...) -> retrying with batch N` | normal: very large inputs (e.g. EfficientNet-B7 at 600 px) are retried with smaller batches automatically. |
| `FAILED <job> three times` | the job is skipped for this session; the rest continues. It is retried on the next run. |
| `ALREADY RUNNING` | the benchmark is open in another window. |
| progress shows only `GPU0` | only one GPU is visible to `nvidia-smi`; the benchmark still runs, one job at a time. |
| GPU at 0 % for long periods | the disk is too slow (HDD). Move the folder to an SSD (stop, move, run again — nothing is lost). |

Logs: `logs/main.log` (overview) and `logs/jobs/<job>.log` (details of each job).
