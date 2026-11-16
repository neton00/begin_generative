# 실행: powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
# 설치만 수행합니다. 과제/학습은 실행하지 않습니다.
# PyTorch 공식 wheel: https://pytorch.org/get-started/previous-versions/
param(
    [ValidateSet('cu128', 'cu126', 'cpu')][string]$Cuda = 'cu128',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe $Arguments" }
}
Invoke-Checked $Python @('-c', 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Use Python 3.10-3.13 for this pinned environment"')
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Checked $Python @('-m', 'venv', '.venv')
}
$TaskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
Invoke-Checked $TaskPython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked $TaskPython @('-m', 'pip', 'install', 'torch==2.11.0', '--index-url', "https://download.pytorch.org/whl/$Cuda")
Invoke-Checked $TaskPython @('-m', 'pip', 'install', '-r', 'requirements-weeks12-13.txt')
# 커널 등록 파일도 프로젝트 안에만 저장합니다.
Invoke-Checked $TaskPython @('-m', 'ipykernel', 'install', '--prefix', (Join-Path $PSScriptRoot '.venv'), '--name', 'generative-gpu', '--display-name', 'Generative assignments (local GPU)')
Write-Host 'Select .venv/Scripts/python.exe in VS Code, or run .venv/Scripts/jupyter-lab.exe.'
Write-Host 'GPU driver/runtime check occurs when you run the notebooks.'

