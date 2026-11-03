param([string]$Environment = '.venv')
$ErrorActionPreference = 'Stop'
$courseRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $courseRoot
$courseEnv = [IO.Path]::GetFullPath((Join-Path $courseRoot $Environment))
if (-not $courseEnv.StartsWith($courseRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Environment must be inside the course repository.' }
if (-not (Test-Path -LiteralPath (Join-Path $courseEnv 'Scripts\python.exe'))) {
    py -3.11 -m venv $courseEnv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.11 environment.' }
}
$coursePython = Join-Path $courseEnv 'Scripts\python.exe'
& $coursePython -m pip install -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $coursePython -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'Course package installation failed.' }
Write-Host ('Environment ready. Run: & "' + $coursePython + '" -m jupyterlab')
