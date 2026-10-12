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
    bash .agent/run_in_container.sh <strategy> <qdrant-url> [skip-second-run]

    Everything is logged to .agent/index-run.log.

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
# ``docker compose up -d`` prints status to stderr even on success.
# Use Continue to capture output, then check $LASTEXITCODE.
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

# 6. Run the container script.
$network = "enterprise-rag-agent_default"
$mountRepo = "${projectRoot}:/rag_agent"
$containerScript = "/rag_agent/.agent/run_in_container.sh"
$skipFlag = if ($SkipSecondRun) { "true" } else { "false" }

Write-Log "===== CONTAINER INDEXING (strategy=$Strategy, skipSecondRun=$skipFlag) ====="
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$dockerOutput = docker run --rm `
    --network $network `
    -v $mountRepo `
    -v hf_cache:/root/.cache/huggingface `
    -v pip_cache:/root/.cache/pip `
    --workdir /rag_agent `
    python:3.11-slim bash "$containerScript" "$Strategy" "http://qdrant:6333" "$skipFlag" 2>&1 | Out-String
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($dockerOutput) { $dockerOutput.Trim() | ForEach-Object { Write-Log "docker: $_" } }
if ($dockerExit -ne 0) {
    Write-Log "ERROR: docker run failed with exit code $dockerExit"
    exit 1
}

Write-Log "===== index_in_docker.ps1 done ====="
