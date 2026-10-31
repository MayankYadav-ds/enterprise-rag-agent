<#
.SYNOPSIS
    Run a search command inside the indexing container.

.DESCRIPTION
    The BAAI/bge-small-en-v1.5 embedding model is backed by PyTorch, which
    Windows Application Control blocks on this machine (WinError 4551 on
    torch/lib/c10.dll). This script runs a search command inside a
    python:3.11-slim container where torch is installed from the CPU wheel
    index, so the real model can load and search the live Qdrant collection.

    Steps: docker compose up -d, wait for Qdrant, then in the container:
    bash scripts/run_search_in_container.sh (with the command passed through
    the SEARCH_COMMAND environment variable).

    Everything is logged to .agent/search-run.log.

    This is the search companion to scripts/index_in_docker.ps1. It does not
    change index_in_docker.ps1 behaviour.

.PARAMETER Command
    The shell command to run inside the container, e.g.
    "python scripts/search.py 'What risks?' --top-k 5"

.EXAMPLE
    .\scripts\run_in_docker.ps1 -Command "python scripts/search.py 'What risks?'"
    .\scripts\run_in_docker.ps1 -Command "python scripts/search.py --queries-file /rag_agent/.agent/queries-live.txt --top-k 5"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Command
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
$logPath = Join-Path $projectRoot ".agent\search-run.log"
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
Write-Log "===== run_in_docker.ps1 start (command=$Command) ====="

# 4. Start Qdrant.
# ``docker compose up -d`` prints status to stderr even on success. Stream
# every line to the log as it is produced (the old Out-String captured the
# whole run and wrote it only after the container exited, so the user saw
# silence for the duration). Success is decided from $LASTEXITCODE, not from
# the presence of error records.
Write-Log "docker compose up -d"
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker compose up -d 2>&1 | ForEach-Object { Write-Log "compose: $_" }
$composeExit = $LASTEXITCODE
$ErrorActionPreference = $prev
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

# 6. Run the command in the container.
#
# The command is a full shell string (e.g. "python scripts/search.py '...'"),
# so it is passed to the container script through SEARCH_COMMAND rather than as
# a single argv element: bash "$containerScript" "$Command" would hand the whole
# string to $0, leaving "$@" empty. The container script reads SEARCH_COMMAND,
# word-splits it with 'eval set --', and execs it, so argv is correct and the
# model is loaded only once for the whole file.
$network = "enterprise-rag-agent_default"
$mountRepo = "${projectRoot}:/rag_agent"
$containerScript = "/rag_agent/scripts/run_search_in_container.sh"

Write-Log "===== CONTAINER SEARCH (command=$Command) ====="
# Inside the container Qdrant is reached as http://qdrant:6333 (the compose
# service name on the shared network), NOT http://localhost:6333, which is
# unreachable from the container's own network namespace. search.py defaults
# to localhost; inject the container-side URL unless the caller already set
# --qdrant-url.
if ($Command -notmatch "(--qdrant-url|-q)\s") {
    $Command = "$Command --qdrant-url http://qdrant:6333"
}
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker run --rm `
    --network $network `
    -v $mountRepo `
    -v hf_cache:/root/.cache/huggingface `
    -v pip_cache:/root/.cache/pip `
    --workdir /rag_agent `
    -e SEARCH_COMMAND="$Command" `
    python:3.11-slim bash "$containerScript" 2>&1 | ForEach-Object { Write-Log "docker: $_" }
$dockerExit = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($dockerExit -ne 0) {
    Write-Log "ERROR: docker run failed with exit code $dockerExit"
    exit 1
}

Write-Log "===== run_in_docker.ps1 done ====="
