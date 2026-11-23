SATELLITE BENCHMARK - HOW TO USE
================================

WHAT IT DOES (automatically, one step after another)
  0. Gets the data: downloads 2 zip files from Hugging Face (about 15.5 GB) - or uses them if
     you put datasets_initail.zip and upload_package.zip next to RUN.bat yourself.
     Builds the CLEAN list (75 classes) and the RAW list (132 classes) from the original images.
  1. Smoke test (short): checks the 4 GPUs, loads all 40 models, runs a mini test.
     Problems show up here, in the first hour - not after days.
  A. 40 models (20 CNN + 20 ViT): linear probe on the CLEAN list, 10 runs each - REAL images only.
  B. Best 5 CNN + best 5 ViT: linear probe on the RAW list, 10 runs each.
  C. Best 2 CNN + best 2 ViT: fine-tuning on the CLEAN list, 3 seeds each - real images + augmented
     copies (training only) + random flips / crops / 90-degree rotations.
  R. Results PDF + tables + t-SNE maps and silhouette scores (clean vs raw labels, same images).
  Augmented images are used for FINE-TUNING TRAINING ONLY, and only when their original image is in
  training. Validation and test always use real images only (no leakage).
  Classes that stayed below 2,000 images after cleaning (center, Park, square) are topped up
  automatically with new augmented copies (each copy has at least one random change).

BEFORE THE FIRST START
  1. Put this folder on the fast disk with a SHORT path, for example  C:\SatBench
     (needs about 150 GB free).
  2. NVIDIA driver must be installed (the command  nvidia-smi  must work).
  3. hf_token.txt  - create this file next to RUN.bat with one line:
         token=hf_xxxxxxxxxxxxxxxxx
     (your Hugging Face token, "Read" is enough). Not needed if you put the two zips here yourself.
  4. notify.txt    - optional: phone messages. Open it and follow the 3 steps inside.
  5. rclone.conf   - optional: Google Drive upload. See "GOOGLE DRIVE" below.

START / STOP / CONTINUE
  Start:     double-click RUN.bat
             (first time it installs its own Python 3.11 + packages into the folder "runtime",
              about 10-20 minutes; nothing is installed on the rest of the PC)
  Stop:      double-click STOP.bat and press Y
  Continue:  double-click RUN.bat again. Finished work is NEVER repeated; fine-tuning continues
             from the last finished epoch.
  Do not start RUN.bat twice at the same time - the second one refuses to run.

HOW LONG (measured on one RTX A6000)
  Setup: 10-20 min (first time only). Data download: depends on the internet (15.5 GB).
  Smoke test: about 20 min.  A: about 6 hours.  B: about 2 hours.  C: about 24 hours.
  With 4 GPUs the jobs run in parallel and the total drops to roughly a third.

PHONE MESSAGES (ntfy)
  You get: started, smoke test passed/failed, progress every hour (% done, time left,
  GPU temperature and use), errors (with the end of the log), finished (with the results PDF).
  If the phone or internet has a problem, the benchmark simply continues.

GOOGLE DRIVE (amansynaug@gmail.com)
  Results and logs are copied to  gdrive:Satellite/<run name>/  every 10 minutes and at the end.
  Model files (*.pt) and the big feature files are NOT uploaded.
  To make rclone.conf (one time, on any PC):
    1. Download rclone from https://rclone.org/downloads/ and open a terminal in its folder.
    2. Run:  rclone config   ->  n (new remote) -> name: gdrive -> storage: drive
       -> leave client id / secret empty -> scope: 1 (full access) -> keep the defaults
       -> log in with amansynaug@gmail.com in the browser -> finish.
    3. Copy the file  rclone.conf  (rclone config file  shows where it is) next to RUN.bat.
  rclone itself is downloaded automatically if rclone.exe is not next to RUN.bat.
  Keep rclone.conf private: it is never uploaded or printed by the benchmark.
  If upload fails, the benchmark keeps running and you get a phone message.

  "401 Invalid Credentials" or "invalid_grant" = the Google login expired.
    Fix: on any PC run   rclone config reconnect gdrive:   (or make a new rclone.conf),
    then replace rclone.conf next to RUN.bat. No restart needed - the next upload uses it.

WHERE ARE THE RESULTS
  results\report\results.pdf             the summary (also sent to the phone at the end)
  results\report\linear_probe_summary.csv all models, clean and raw
  results\report\finetune_summary.csv     fine-tuning, mean and SD over seeds
  results\report\clean_vs_raw.csv         same models on raw vs clean labels
  results\report\stats.json               CNN vs ViT test, Kendall tau, silhouette summary
  results\report\silhouette_clean_vs_raw.csv  how well clean vs raw labels match the features
  results\report\tsne_<model>.png           t-SNE maps (best CNN and best ViT), 300 dpi
  results\<job>\result.json                every single job in detail
  logs\main.log  and  logs\jobs\*.log       what happened

IF SOMETHING GOES WRONG
  "SETUP FAILED"        read logs\setup.log. Usually internet. Run RUN.bat again.
                        Still failing: delete the folder "runtime" and run RUN.bat again.
  "Smoke test FAILED"   read logs\jobs\smoke.log - nothing else was started, nothing is lost.
  "PyTorch cannot use the GPUs"   update the NVIDIA driver, delete "runtime", run again.
  A model could not load  it is skipped automatically and listed in the phone message.
  Out of GPU memory     handled automatically (smaller batch).
  A job fails 3 times   it is skipped for now; the rest continues; it is tried again on the next RUN.
  Hugging Face 401/403  check hf_token.txt (token=hf_...), or put the two zips next to RUN.bat.

SETTINGS (optional)  settings.ini  - number of runs, seeds, epochs, Drive folder name.
