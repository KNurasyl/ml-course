$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$coursePython = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
& $coursePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $coursePython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $coursePython tools/validate_course.py --require-local
if ($LASTEXITCODE -ne 0) { throw 'Course integrity check failed' }
& $coursePython tools/run_labs.py
if ($LASTEXITCODE -ne 0) { throw 'Module laboratories failed' }
& $coursePython tools/run_extended_labs.py
if ($LASTEXITCODE -ne 0) { throw 'Extended experiments or student-contract checks failed' }
& $coursePython tools/check_lesson_examples.py
if ($LASTEXITCODE -ne 0) { throw 'Authored Python examples failed' }
& $coursePython tools/run_notebooks.py --kind all
if ($LASTEXITCODE -ne 0) { throw 'Notebook execution failed' }
Write-Host 'Technical checks passed. Learner mastery and the human pilot are not assessed by this script.'
