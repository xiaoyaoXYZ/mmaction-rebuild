$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$experimentPython = Join-Path $experimentRoot '.venv\Scripts\python.exe'
$runner = Join-Path $experimentRoot 'scripts\run_baseline.py'
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$smokeOutput = Join-Path $experimentRoot "work_dirs\smoke_$runStamp"
$trainOutput = Join-Path $experimentRoot "work_dirs\train_$runStamp"
$evalOutput = Join-Path $experimentRoot "work_dirs\evaluate_$runStamp"

if (-not (Test-Path -LiteralPath $experimentPython)) {
    throw 'The isolated environment is missing. See README.md for setup.'
}
& $experimentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $experimentPython $runner smoke --output-dir $smokeOutput
if ($LASTEXITCODE -ne 0) { throw "GPU smoke failed; see $smokeOutput" }
& $experimentPython $runner train --output-dir $trainOutput
if ($LASTEXITCODE -ne 0) { throw "Training failed; see $trainOutput" }
& $experimentPython $runner evaluate --checkpoint (Join-Path $trainOutput 'epoch_1.pth') --output-dir $evalOutput
if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation failed; see $evalOutput" }
Write-Output "Completed. Training: $trainOutput ; evaluation: $evalOutput"
