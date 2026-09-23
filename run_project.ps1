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
    Write-Warning 'MLP checkpoint is missing. The API will start, but MLP predictions will return 503 until trained.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'models\fraud_detector_ieee.pt'))) {
    Write-Warning 'IEEE-CIS checkpoint is missing. IEEE-CIS predictions will return 503 until trained with the local IEEE-CIS files.'
}

$frontendReady = Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\node_modules\vite\bin\vite.js')
$streamlitReady = $false
& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -eq 0) { $streamlitReady = $true }

$apiProcess = Start-Process -FilePath 'powershell.exe' -WorkingDirectory $PSScriptRoot -PassThru -ArgumentList @(
    '-NoExit', '-Command', "& '$python' -m uvicorn api.app:app --host 127.0.0.1 --port 8000"
)

$viteProcess = $null
if ($frontendReady) {
    $viteProcess = Start-Process -FilePath 'powershell.exe' -WorkingDirectory (Join-Path $PSScriptRoot 'frontend') -PassThru -ArgumentList @(
        '-NoExit', '-Command', 'npm.cmd run dev'
    )
} else {
    Write-Warning 'Frontend dependencies are missing. Run npm.cmd ci in frontend before starting Vite.'
}

if ($streamlitReady) {
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $PSScriptRoot -ArgumentList @(
        '-NoExit', '-Command', "& '$python' -m streamlit run streamlit_app.py --server.port 8501 --server.headless true"
    )
} else {
    Write-Warning 'Streamlit is not installed in .venv. The Streamlit dashboard was skipped.'
}

function Wait-ForHttpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    return $false
}

Write-Host 'FastAPI: http://127.0.0.1:8000'
if ($streamlitReady) { Write-Host 'Streamlit: http://localhost:8501' }

if (-not (Wait-ForHttpEndpoint -Uri 'http://127.0.0.1:8000/health')) {
    throw "FastAPI did not become reachable at http://127.0.0.1:8000. Check the backend window (PID $($apiProcess.Id))."
}

if ($frontendReady) {
    Write-Host 'React/Vite: http://localhost:5173'
    if (-not (Wait-ForHttpEndpoint -Uri 'http://localhost:5173')) {
        throw "React/Vite did not become reachable at http://localhost:5173. Check the frontend window (PID $($viteProcess.Id))."
    }

    Start-Process -FilePath 'http://localhost:5173'
} else {
    Write-Warning 'React/Vite was not started, so no browser page was opened.'
}
