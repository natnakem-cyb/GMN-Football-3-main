$ErrorActionPreference = 'Stop'

$scenario = 'academy_3_vs_1_with_keeper'
$timesteps = 200000
$seeds = @(123, 999)
$bridgePort = 5050
$py = 'C:\Python314\python.exe'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $repoRoot 'training\results\forensics'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Wait-PortFree {
    param([int]$Port, [string]$Label)
    Write-Output "[orchestrator] Waiting for port $Port to be free ($Label)..."
    while ($true) {
        $inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $inUse) { return }
        Start-Sleep -Seconds 15
    }
}

function Invoke-Seed {
    param([int]$Seed)
    $ckpt = "mappo_${scenario}_seed${Seed}_quarantine.pt"
    $log = Join-Path $logDir "train_seed${Seed}.log"
    Write-Output "[orchestrator] Starting seed $Seed (log: $log)"
    $env:GMN_FORENSIC_DEBUG = '0'
    & $py -m training.train_mappo --scenario $scenario --seed $Seed --timesteps $timesteps `
        --n-envs 1 --checkpoint-name $ckpt *>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "seed $Seed training failed with exit code $LASTEXITCODE" }
    Write-Output "[orchestrator] Seed $Seed training complete"
}

# Wait for any existing run (seed 42) to release the bridge port
Wait-PortFree -Port $bridgePort -Label 'prior seed'

foreach ($s in $seeds) {
    Wait-PortFree -Port $bridgePort -Label "before seed $s"
    Invoke-Seed -Seed $s
}

Write-Output '[orchestrator] All seeds trained. Starting evals...'

# Eval each seed's best checkpoint
foreach ($s in @(42) + $seeds) {
    $best = Join-Path $repoRoot "training\models\mappo_${scenario}_seed${s}_best.pt"
    if (-not (Test-Path $best)) { Write-Output "[orchestrator] WARNING: $best not found, skipping eval"; continue }
    $elog = Join-Path $logDir "eval_seed${s}.log"
    Write-Output "[orchestrator] Evaluating seed $s ($best)"
    Wait-PortFree -Port $bridgePort -Label "before eval seed $s"
    & $py (Join-Path $repoRoot 'training\eval_mappo.py') --checkpoint $best --episodes 50 --deterministic *>&1 | Tee-Object -FilePath $elog
}

Write-Output '[orchestrator] Running exploit suite...'
Wait-PortFree -Port $bridgePort -Label 'before exploit tests'
& $py -m pytest (Join-Path $repoRoot 'training\tests\test_reward_exploits.py') -q *>&1 | Tee-Object -FilePath (Join-Path $logDir 'exploit_suite.log')

Write-Output '[orchestrator] DONE. All seeds trained, evaluated, exploit-tested.'
