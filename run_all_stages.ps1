<#
.SYNOPSIS
    Runs the whole pipeline over a folder of point clouds: TupiSAT
    segmentation, PointsToWood wood/leaf, and the forest metrics that use it.

.DESCRIPTION
    Stage 1 is a single container that batches every cloud internally and is
    resumable on its own. Stages 2 and 3 are not inside that container yet
    (the two projects need incompatible Python/PyTorch versions -- see
    runme.md section 0), so this script loops them per plot.

    Every stage skips work that is already finished, so re-running after an
    interruption continues rather than restarting. Use -Force to redo.

.EXAMPLE
    .\run_all_stages.ps1
    .\run_all_stages.ps1 -SkipStage1        # segmentation already done
    .\run_all_stages.ps1 -Only P01,P02      # just those two plots
#>
param(
    [string] $Root       = "E:\GITHUB\SegmentAnyTree",
    [string] $InputDir   = "E:\GITHUB\SegmentAnyTree\data\03-Clipped16m",
    [string] $SatOut     = "E:\GITHUB\SegmentAnyTree\data\04-OUTPUT",
    [string] $PwoodDir   = "E:\GITHUB\SegmentAnyTree\data\05-PWOOD",
    [string] $MetricsDir = "E:\GITHUB\SegmentAnyTree\data\06-METRICS",
    [string] $ReportDir  = "E:\GITHUB\SegmentAnyTree\data\07-RELATORIO",
    [string] $SatImage   = "tupisat:latest",
    [string] $PtwImage   = "pointstowood:latest",
    [string] $Region     = "eu",
    [ValidateSet("en","pt","es")]
    [string] $Lang       = "en",
    [switch] $SkipStage1,
    [switch] $SkipStage2,
    [switch] $SkipStage3,
    [switch] $SkipStage4,
    [switch] $Force,
    [string[]] $Only
)

$ErrorActionPreference = "Stop"

function Say($msg, $color = "White") {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -ForegroundColor $color
}

function Wait-Container($name) {
    # docker wait blocks until the container exits and prints its exit code.
    $code = (docker wait $name | Select-Object -Last 1)
    return [int]$code
}

# $SatOut is created here too: it is a bind-mount target for stage 1, and the
# discovery below reads it even when stage 1 is skipped.
New-Item -ItemType Directory -Force -Path $SatOut, $PwoodDir, $MetricsDir, $ReportDir | Out-Null

# ---------------------------------------------------------------- stage 1
if (-not $SkipStage1) {
    Say "STAGE 1  TupiSAT segmentation over $InputDir" "Cyan"

    # The image bakes the code in (stages 3-4 mount the repo instead), so a
    # stale image silently produces the old crown rule and uncorrected
    # diameters. Refuse to run rather than hand back results that look fine.
    $probe = docker run --rm --entrypoint bash $SatImage -c `
        "grep -c crown_wood_frac_threshold /home/nibio/mutable-outside-world/tupisat_inference/forest_metrics/config.py"
    if ([int]($probe | Select-Object -Last 1) -lt 1) {
        throw "$SatImage does not contain the current code. Rebuild first: docker build -f Dockerfile.pandas-fix -t $SatImage ."
    }

    docker rm -f tupisat_all 2>$null | Out-Null
    $forceFlag = @()
    if ($Force) { $forceFlag = @("--force") }

    docker run -d --gpus all --name tupisat_all `
        --mount "type=bind,source=$InputDir,target=/home/nibio/mutable-outside-world/bucket_in_folder" `
        --mount "type=bind,source=$SatOut,target=/home/nibio/mutable-outside-world/bucket_out_folder" `
        $SatImage @forceFlag | Out-Null

    Say "  container started; follow with: docker logs -f tupisat_all"
    $code = Wait-Container "tupisat_all"
    if ($code -ne 0) { throw "Stage 1 exited with $code -- see: docker logs tupisat_all" }
    Say "  stage 1 done" "Green"
}

# Each finished plot is a <stem>_SAT_output folder holding <stem>.laz. Older
# runs wrote the cloud as <stem>_crown_classified.laz instead; accept both so
# this works against output produced before the in-place IsCrown change.
$plots = @()
foreach ($dir in Get-ChildItem -Path $SatOut -Directory -Filter "*_SAT_output") {
    $stem = $dir.Name -replace "_SAT_output$", ""
    $cloud = Join-Path $dir.FullName "$stem.laz"
    if (-not (Test-Path $cloud)) { $cloud = Join-Path $dir.FullName "${stem}_crown_classified.laz" }
    if (-not (Test-Path $cloud)) {
        Say "  ! no point cloud in $($dir.Name), skipping" "Yellow"
        continue
    }
    if ($Only -and -not ($Only | Where-Object { $stem -like "*$_*" })) { continue }
    $plots += [pscustomobject]@{ Stem = $stem; Cloud = $cloud }
}
Say ("{0} plot(s) to process" -f $plots.Count) "Cyan"

# ---------------------------------------------------------------- stage 2
if (-not $SkipStage2) {
    Say "STAGE 2  PointsToWood wood/leaf" "Cyan"
    $i = 0
    foreach ($p in $plots) {
        $i++
        $out = Join-Path $PwoodDir "$($p.Stem)_pwood.laz"
        if ((Test-Path $out) -and -not $Force) {
            Say ("  [{0}/{1}] {2}  already done, skipping" -f $i, $plots.Count, $p.Stem)
            continue
        }
        Say ("  [{0}/{1}] {2}  ~7 min" -f $i, $plots.Count, $p.Stem)
        docker rm -f ptw_run 2>$null | Out-Null

        # expandable_segments and --tta 1 are not tuning: without them this
        # stalls silently on a 16 GB card, with no exception and the
        # container still "Up". See runme.md section 2.
        docker run -d --gpus all --name ptw_run `
            -v "${Root}\data:/app/sat_data" `
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True `
            --entrypoint python $PtwImage preinstance_pipeline.py `
                ($p.Cloud -replace [regex]::Escape("$Root\data"), "/app/sat_data" -replace "\\", "/") `
                --preinstance-field PredInstance --region $Region `
                --memory-fraction 0.45 --tta 1 --no-ptw-output `
                --output ("/app/sat_data/05-PWOOD/$($p.Stem)_pwood.laz") | Out-Null

        $code = Wait-Container "ptw_run"
        if ($code -ne 0) {
            Say ("  ! {0} failed (exit {1}); continuing with the rest" -f $p.Stem, $code) "Red"
            docker logs --tail 20 ptw_run
            continue
        }
        Say ("  [{0}/{1}] {2}  ok" -f $i, $plots.Count, $p.Stem) "Green"
    }
}

# ---------------------------------------------------------------- stage 3
if (-not $SkipStage3) {
    Say "STAGE 3  forest metrics with the wood/leaf crown rule" "Cyan"
    $i = 0
    foreach ($p in $plots) {
        $i++
        $pwood = Join-Path $PwoodDir "$($p.Stem)_pwood.laz"
        if (-not (Test-Path $pwood)) {
            Say ("  [{0}/{1}] {2}  no _pwood.laz, skipping" -f $i, $plots.Count, $p.Stem) "Yellow"
            continue
        }
        $done = Join-Path $MetricsDir "$($p.Stem)_tree_metrics.csv"
        if ((Test-Path $done) -and -not $Force) {
            Say ("  [{0}/{1}] {2}  already done, skipping" -f $i, $plots.Count, $p.Stem)
            continue
        }
        Say ("  [{0}/{1}] {2}" -f $i, $plots.Count, $p.Stem)

        # Mounts the repo, so this always runs the code on disk -- no rebuild
        # needed after editing forest_metrics/.
        docker run --rm `
            -v "${Root}:/w" -w /w -e PYTHONPATH=/w `
            --entrypoint python3.8 $SatImage `
            tupisat_inference/forest_metrics/forest_metrics.py `
                --input-las ("data/05-PWOOD/$($p.Stem)_pwood.laz") `
                --output-dir "data/06-METRICS" `
                --stem $p.Stem --verbose
        if ($LASTEXITCODE -ne 0) {
            Say ("  ! {0} failed; continuing" -f $p.Stem) "Red"
            continue
        }
        Say ("  [{0}/{1}] {2}  ok" -f $i, $plots.Count, $p.Stem) "Green"
    }
}

# ---------------------------------------------------------------- stage 4
if (-not $SkipStage4) {
    Say "STAGE 4  per-tree validation pages ($Lang)" "Cyan"
    $i = 0
    foreach ($p in $plots) {
        $i++
        $pwood = Join-Path $PwoodDir "$($p.Stem)_pwood.laz"
        $metrics = Join-Path $MetricsDir "$($p.Stem)_tree_metrics.csv"
        if (-not (Test-Path $pwood) -or -not (Test-Path $metrics)) {
            Say ("  [{0}/{1}] {2}  metrics missing, skipping" -f $i, $plots.Count, $p.Stem) "Yellow"
            continue
        }
        $nTrees = (Import-Csv $metrics | Measure-Object).Count
        $existing = @(Get-ChildItem -Path $ReportDir -Filter "$($p.Stem)_tree*.png" -ErrorAction SilentlyContinue).Count
        if ($existing -ge $nTrees -and -not $Force) {
            Say ("  [{0}/{1}] {2}  {3} pages already there, skipping" -f $i, $plots.Count, $p.Stem, $existing)
            continue
        }
        Say ("  [{0}/{1}] {2}  {3} trees, ~{4} min" -f $i, $plots.Count, $p.Stem, $nTrees, [math]::Ceiling($nTrees * 0.5))

        # The cloud is read and the DTM built once for the whole plot, so
        # --all-trees costs far less than one invocation per tree.
        docker run --rm `
            -v "${Root}:/w" -w /w -e PYTHONPATH=/w -e MPLCONFIGDIR=/tmp/mpl `
            --entrypoint python3.8 $SatImage `
            tupisat_inference/forest_metrics/tree_report.py `
                --input-las ("data/05-PWOOD/$($p.Stem)_pwood.laz") `
                --metrics-dir "data/06-METRICS" --stem $p.Stem `
                --all-trees --output-dir "data/07-RELATORIO" --dpi 200 --lang $Lang
        if ($LASTEXITCODE -ne 0) {
            Say ("  ! {0} failed; continuing" -f $p.Stem) "Red"
            continue
        }
        Say ("  [{0}/{1}] {2}  ok" -f $i, $plots.Count, $p.Stem) "Green"
    }
}

Say "all done" "Green"
Say "  segmented clouds : $SatOut"
Say "  wood/leaf clouds : $PwoodDir"
Say "  metrics          : $MetricsDir"
Say "  report pages     : $ReportDir"
