$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found. Create it and install requirements.txt before running this script."
}

& $python -c "import fastapi, uvicorn, pydantic, torch, pandas, numpy, sklearn, matplotlib, seaborn, imblearn"
if ($LASTEXITCODE -ne 0) {
    throw "Backend dependencies are missing from .venv. Install with: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'models\fraud_detector.pt'))) {
    Write-Warning 'Synthetic MLP checkpoint is missing. The API will start, but synthetic predictions will return 503 until trained.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'models\fraud_detector_ulb.pt'))) {
    Write-Warning 'ULB checkpoint is missing. ULB predictions will return 503 until trained with the external dataset.'
}

$frontendReady = Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\node_modules\vite\bin\vite.js')
$streamlitReady = $false
& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -eq 0) { $streamlitReady = $true }

Start-Process -FilePath 'powershell.exe' -WorkingDirectory $PSScriptRoot -ArgumentList @(
    '-NoExit', '-Command', "& '$python' -m uvicorn api.app:app --host 127.0.0.1 --port 8000"
)

if ($frontendReady) {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory (Join-Path $PSScriptRoot 'frontend') -ArgumentList @(
        '-NoExit', '-Command', 'npm.cmd run dev'
    )
} else {
    Write-Warning 'Frontend dependencies are missing. Run npm.cmd ci in frontend before starting Vite.'
}

if ($streamlitReady) {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $PSScriptRoot -ArgumentList @(
        '-NoExit', '-Command', "& '$python' -m streamlit run streamlit_app.py --server.port 8501"
    )
} else {
    Write-Warning 'Streamlit is not installed in .venv. The Streamlit dashboard was skipped.'
}

Write-Host 'FastAPI: http://127.0.0.1:8000'
if ($frontendReady) { Write-Host 'React/Vite: http://localhost:5173' }
if ($streamlitReady) { Write-Host 'Streamlit: http://localhost:8501' }
