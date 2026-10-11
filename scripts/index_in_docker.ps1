<#
.SYNOPSIS
    Start Qdrant and index the PDFs inside a Linux container.

.DESCRIPTION
    The BAAI/bge-small-en-v1.5 embedding model is backed by PyTorch, which
    Windows Application Control blocks on this machine (WinError 4551 on
    torch/lib/c10.dll). This script runs the whole indexing pipeline inside a
    python:3.11-slim container where torch is installed from the CPU wheel
    index, so the real model can load.

    Steps: docker compose up -d, wait for Qdrant, then in the container:
    pip install torch (CPU) -> pip install -e ".[embeddings]" ->
    index_documents --recreate -> index_documents (no --recreate) ->
    smoke_queries.py. Everything is logged to .agent/index-run.log.

.PARAMETER SkipSecondRun
    Skip the second, idempotent indexing run.

.PARAMETER Strategy
    Chunking strategy. Default: recursive.
#>
[CmdletBinding()]
param(
    [switch]$SkipSecondRun,
    [string]$Strategy = "recursive"
)

$ErrorActionPreference = "Stop"

# 1. Project root from this script's location, so the script works from any folder.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot
Write-Host "Project root: $projectRoot"

# 2. pyproject.toml must exist.
if (-not (Test-Path (Join-Path $projectRoot "pyproject.toml"))) {
    Write-Error "pyproject.toml not found in $projectRoot. Run this script from the repo root or its scripts/ folder."
    exit 1
}

# 3. Logging target.
$logPath = Join-Path $projectRoot ".agent\index-run.log"
$logDir = Split-Path $logPath
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }

function Write-Log {
    param([string]$Line)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Encoding UTF8 -Value "$ts $Line"
    Write-Host "$ts $Line"
}

# Fresh log for this run.
Remove-Item -Path $logPath -ErrorAction SilentlyContinue
Write-Log "===== index_in_docker.ps1 start (strategy=$Strategy, skipSecondRun=$SkipSecondRun) ====="

# 4. Start Qdrant.
#
# ``docker compose up -d`` prints "Container ... Running" to stderr even on
# success, and PowerShell promotes stderr lines into terminating error records
# under ``-ErrorAction Stop``. Temporarily continue on error so the exit code
# (not the exception) is what decides success.
Write-Log "docker compose up -d"
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$composeOutput = docker compose up -d 2>&1 | Out-String
$composeExit = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($composeOutput) { $composeOutput.Trim() | ForEach-Object { Write-Log "compose: $_" } }
if ($composeExit -ne 0) {
    Write-Log "ERROR: docker compose up -d failed with exit code $composeExit"
    exit 1
}

# 5. Wait until Qdrant answers on the collections endpoint.
$deadline = (Get-Date).AddMinutes(2)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:6333/collections" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        # Qdrant is not ready yet; retry.
    }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    Write-Log "ERROR: Qdrant did not answer on http://localhost:6333/collections"
    exit 1
}
Write-Log "Qdrant ready on http://localhost:6333/collections"

# 6. Indexing commands, run inside the Linux container on the compose network.
#    The repo is mounted at /rag_agent, so data/raw and data/processed are the
#    host's. Qdrant is reachable as http://qdrant:6333 because both containers
#    share the enterprise-rag-agent_default network. Anonymous volumes cache the
#    Hugging Face model and pip downloads between runs.
$network = "enterprise-rag-agent_default"
$mountRepo = "${projectRoot}:/rag_agent"

function Invoke-DockerIndex {
    param([string]$Label, [string]$Command)
    Write-Log "===== $Label ====="
    Write-Log "docker run --rm --network $network python:3.11-slim bash -lc ..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $dockerOutput = docker run --rm `
        --network $network `
        -v $mountRepo `
        -v hf_cache:/root/.cache/huggingface `
        -v pip_cache:/root/.cache/pip `
        --workdir /rag_agent `
        python:3.11-slim bash -lc $Command 2>&1 | Out-String
    $dockerExit = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($dockerOutput) { $dockerOutput.Trim() | ForEach-Object { Write-Log "docker: $_" } }
    if ($dockerExit -ne 0) {
        Write-Log "ERROR: docker run failed with exit code $dockerExit"
        exit 1
    }
}

# RUN 1: recreate the collection and index from scratch.
$run1Cmd = @'
set -eu
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --quiet -e ".[embeddings]"
python scripts/index_documents.py --input data/raw --strategy __STRATEGY__ --recreate --qdrant-url http://qdrant:6333
python scripts/smoke_queries.py --collection rag_chunks --qdrant-url http://qdrant:6333
'@
$run1Cmd = $run1Cmd -replace "__STRATEGY__", $Strategy
Invoke-DockerIndex "RUN 1 (recreate)" $run1Cmd

# RUN 2: idempotent re-index, must upsert nothing.
if ($SkipSecondRun) {
    Write-Log "SKIP: second run (-SkipSecondRun)"
} else {
    $run2Cmd = @'
set -eu
python -m pip install --quiet -e ".[embeddings]"
python scripts/index_documents.py --input data/raw --strategy __STRATEGY__ --qdrant-url http://qdrant:6333
python scripts/smoke_queries.py --collection rag_chunks --qdrant-url http://qdrant:6333
'@
    $run2Cmd = $run2Cmd -replace "__STRATEGY__", $Strategy
    Invoke-DockerIndex "RUN 2 (idempotent)" $run2Cmd
}

Write-Log "===== index_in_docker.ps1 done ====="
