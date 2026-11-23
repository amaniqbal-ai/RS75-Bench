# setup.ps1 - called by RUN.bat. First time: installs a private Python 3.11 and the pinned
# packages into the folder "runtime" (nothing is installed system-wide). Later runs: only checks.
# "Continue": in Windows PowerShell 5.1, "Stop" turns harmless pip warnings on stderr into fatal errors.
# Every important step is checked explicitly instead ($LASTEXITCODE / Test-Path / -ErrorAction Stop).
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$rt = Join-Path $here "runtime"
$logs = Join-Path $here "logs"
New-Item -ItemType Directory -Force -Path $rt, $logs | Out-Null
try { Start-Transcript -Path (Join-Path $logs "setup.log") -Append | Out-Null } catch {}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
$MARKER = Join-Path $rt "packages_ok_v1.txt"
$PYPATH = Join-Path $rt "python_path.txt"

function Say($m) { Write-Host ("[setup] " + $m) }
function Fail($m) { Write-Host ""; Write-Host ("SETUP PROBLEM: " + $m) -ForegroundColor Red; try { Stop-Transcript | Out-Null } catch {}; exit 1 }

function Download($url, $dest) {
    for ($i = 1; $i -le 3; $i++) {
        try {
            $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
            if ($curl) {
                & curl.exe -L -f -s -S -A $UA --retry 3 -o $dest $url
                if ($LASTEXITCODE -eq 0 -and (Test-Path $dest) -and (Get-Item $dest).Length -gt 100000) { return $true }
            }
            Invoke-WebRequest -Uri $url -OutFile $dest -UserAgent $UA -UseBasicParsing -ErrorAction Stop
            if ((Test-Path $dest) -and (Get-Item $dest).Length -gt 100000) { return $true }
        } catch { Say ("download attempt $i failed: " + $_.Exception.Message) }
        Start-Sleep -Seconds (10 * $i)
    }
    return $false
}

if ($here.Length -gt 60) { Say "WARNING: the folder path is long ($($here.Length) characters). A short path like C:\SatBench avoids Windows path-length problems." }

# ---------------------------------------------------------------- 1. Python 3.11
$py = $null
if (Test-Path $PYPATH) {
    $p = (Get-Content $PYPATH -Raw).Trim()
    if (Test-Path $p) { $py = $p }
}
if (-not $py) {
    # the folder may have been MOVED (e.g. from D: to C:): reuse the Python already inside "runtime"
    foreach ($cand in @((Join-Path $rt "python_nuget\tools\python.exe"), (Join-Path $rt "python_inst\python.exe"))) {
        if (Test-Path $cand) { $py = $cand; Set-Content -Path $PYPATH -Value $py -Encoding ASCII; Say "Folder was moved - reusing Python at $py"; break }
    }
}
if (-not $py) {
    Say "Installing a private Python 3.11 (one time, about 1 minute)..."
    $nupkg = Join-Path $rt "python-3.11.9.zip"
    $dir = Join-Path $rt "python_nuget"
    if (Download "https://www.nuget.org/api/v2/package/python/3.11.9" $nupkg) {
        try {
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir -ErrorAction Stop }
            Expand-Archive -Path $nupkg -DestinationPath $dir -Force -ErrorAction Stop
            $cand = Join-Path $dir "tools\python.exe"
            if (Test-Path $cand) { $py = $cand }
        } catch { Say ("could not unpack the NuGet Python: " + $_.Exception.Message) }
    }
    if (-not $py) {
        Say "NuGet Python not available - trying the official installer..."
        $exe = Join-Path $rt "python-3.11.9-amd64.exe"
        $dir2 = Join-Path $rt "python_inst"
        if (-not (Download "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" $exe)) {
            Fail "Could not download Python. Check the internet connection and run RUN.bat again."
        }
        $instArgs = "/quiet InstallAllUsers=0 TargetDir=`"$dir2`" Include_launcher=0 PrependPath=0 Shortcuts=0 AssociateFiles=0 Include_test=0 Include_doc=0"
        Start-Process -FilePath $exe -ArgumentList $instArgs -Wait
        $cand = Join-Path $dir2 "python.exe"
        if (Test-Path $cand) { $py = $cand } else { Fail "Python installation did not finish. See logs\setup.log." }
    }
    Set-Content -Path $PYPATH -Value $py -Encoding ASCII
}
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -ne "3.11") { Fail "Expected Python 3.11 but found $ver at $py. Delete the folder 'runtime' and run again." }
Say "Python: $py"

# ---------------------------------------------------------------- 2. packages (pinned)
if (-not (Test-Path $MARKER)) {
    Say "Installing packages (one time, about 5-15 minutes, ~3 GB for PyTorch with CUDA)..."
    & $py -m ensurepip --upgrade 2>&1 | Out-Host
    & $py -m pip install --upgrade --no-warn-script-location "pip==24.2" 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "pip could not be prepared. See logs\setup.log." }
    & $py -m pip install --no-warn-script-location "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch (CUDA 12.1) could not be installed. Check the internet and run RUN.bat again." }
    & $py -m pip install --no-warn-script-location "numpy==1.26.4" "pandas==2.2.2" "scikit-learn==1.5.1" "scipy==1.13.1" `
        "pillow==10.4.0" "matplotlib==3.9.2" "timm==1.0.9" "safetensors==0.4.5" "psutil==6.0.0" "tqdm==4.66.5" `
        "requests==2.32.3" "huggingface_hub>=0.30,<1.0" 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "Some packages could not be installed. See logs\setup.log." }
    & $py -c "import torch, torchvision, timm, numpy, pandas, sklearn, scipy, matplotlib, PIL, psutil, huggingface_hub; print('packages OK, torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())" 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "Packages were installed but cannot be imported. Delete the folder 'runtime' and run again." }
    Set-Content -Path $MARKER -Value (Get-Date).ToString() -Encoding ASCII
} else {
    Say "Packages already installed."
}

# ---------------------------------------------------------------- 3. GPU check (warning only; main.py stops cleanly if needed)
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) { & nvidia-smi.exe -L 2>&1 | Out-Host } else { Say "WARNING: nvidia-smi not found - is the NVIDIA driver installed?" }
try { Stop-Transcript | Out-Null } catch {}
exit 0
