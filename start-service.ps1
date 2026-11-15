$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$model = Join-Path $PSScriptRoot 'artifacts\run-42'
$backendEnv = Join-Path $PSScriptRoot '..\abc-backend\.env'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python environment not found. Create .venv and install requirements.txt first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $model 'metadata.json'))) {
    throw 'Model not found at artifacts\run-42. Generate a model as described in README.md.'
}
if (-not (Test-Path -LiteralPath $backendEnv)) {
    throw 'Backend .env not found. Configure FORECAST_RESEARCH_TOKEN first.'
}

$matches = @(Select-String -LiteralPath $backendEnv -Pattern '^\s*FORECAST_RESEARCH_TOKEN\s*=\s*(.*?)\s*$')
if ($matches.Count -ne 1) {
    throw 'Backend .env must contain exactly one FORECAST_RESEARCH_TOKEN.'
}
$token = $matches[0].Matches[0].Groups[1].Value.Trim()
if ($token.Length -ge 2 -and (($token.StartsWith('"') -and $token.EndsWith('"')) -or
    ($token.StartsWith("'") -and $token.EndsWith("'")))) {
    $token = $token.Substring(1, $token.Length - 2)
}
if ($token -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'FORECAST_RESEARCH_TOKEN must be nonempty and contain only letters, digits, underscores or hyphens.'
}

$env:MODEL_DIR = (Resolve-Path -LiteralPath $model).Path
$env:FORECAST_API_TOKEN = $token
Push-Location $PSScriptRoot
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $python -m uvicorn service:app --host 127.0.0.1 --port 8001
    if ($LASTEXITCODE -ne 0) {
        throw "Forecast service exited with code $LASTEXITCODE."
    }
}
finally {
    $ErrorActionPreference = $previousPreference
    Pop-Location
}
