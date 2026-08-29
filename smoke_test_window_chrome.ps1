#Requires -Version 5.1
<#
    Runs smoke_test_window_chrome.py on a Windows machine that has nothing on
    it yet: installs Python and Git if they are missing, fetches the repo,
    builds a venv with pywebview, and starts the check. Everything printed is
    also written to a transcript, so the result is one file to hand back.

    Nothing here needs administrator rights, and nothing outside the two paths
    below is touched.

        powershell -ExecutionPolicy Bypass -File .\smoke_test_window_chrome.ps1

    Re-running is cheap: the tools, the clone and the venv are all reused.
#>

$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/AnyLearning/anylearning'
$Root = Join-Path $HOME 'anylearning-test'
$LogFile = Join-Path $HOME 'anylearning-window-check.txt'

function Write-Step($message) {
    Write-Host ''
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Update-SessionPath {
    # winget updates PATH for new processes, not for this one.
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Install-IfMissing($command, $wingetId, $name) {
    if (Get-Command $command -ErrorAction SilentlyContinue) {
        Write-Host "$name is already here"
        return
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "$name is missing and winget is not available. Install $name by hand, then re-run this."
    }
    Write-Step "Installing $name"
    winget install -e --id $wingetId -h --accept-source-agreements --accept-package-agreements
    Update-SessionPath
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$name installed but is not on PATH yet. Open a new PowerShell window and re-run this."
    }
}

Start-Transcript -Path $LogFile -Force | Out-Null

try {
    Write-Step 'Machine'
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Host ("{0} (build {1}), {2}" -f $os.Caption, $os.BuildNumber, $env:PROCESSOR_ARCHITECTURE)

    # 96 is 100%, 120 is 125%, 144 is 150%. The drag rectangles the page
    # reports are scaled by this, so it belongs in the report.
    $dpi = (Get-ItemProperty 'HKCU:\Control Panel\Desktop\WindowMetrics' -Name AppliedDPI -ErrorAction SilentlyContinue).AppliedDPI
    if ($dpi) {
        Write-Host ("Display scaling: {0} DPI ({1}%)" -f $dpi, [math]::Round($dpi / 96 * 100))
    }
    else {
        Write-Host 'Display scaling: not recorded (100%)'
    }

    Write-Step 'WebView2 runtime'
    $webview2 = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\EdgeWebView\Application'),
        (Join-Path $env:ProgramFiles 'Microsoft\EdgeWebView\Application')
    ) | Where-Object { $_ -and (Test-Path $_) }
    if ($webview2) {
        Write-Host "Found at $($webview2[0])"
    }
    else {
        Write-Host 'Not found; installing (the window cannot render without it)'
        Install-IfMissing 'msedgewebview2' 'Microsoft.EdgeWebView2Runtime' 'WebView2 runtime'
    }

    Write-Step 'Tools'
    # 3.13 is what the project is built and tested on.
    Install-IfMissing 'py' 'Python.Python.3.13' 'Python'
    Install-IfMissing 'git' 'Git.Git' 'Git'

    Write-Step 'Source'
    # Run from inside a checkout and that checkout is used as-is; otherwise
    # clone one. The clone will ask you to sign in to GitHub -- the repo is
    # private -- and brings the built frontend with it.
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot 'smoke_test_window_chrome.py'))) {
        $Root = $PSScriptRoot
        Write-Host "Using the checkout at $Root"
    }
    elseif (Test-Path (Join-Path $Root '.git')) {
        Write-Host "Updating $Root"
        git -C $Root pull --ff-only
    }
    else {
        Write-Host "Cloning into $Root"
        git clone --depth 1 $RepoUrl $Root
    }

    $checkScript = Join-Path $Root 'smoke_test_window_chrome.py'
    if (-not (Test-Path $checkScript)) {
        throw "smoke_test_window_chrome.py is not in $Root. Is this the right branch?"
    }

    # The built frontend is not in the repository -- it is a build artefact --
    # so a fresh clone has no window to show. Build it, which needs Node.
    $export = Join-Path $Root 'anylearning\frontend-dist'
    if (-not (Test-Path $export)) {
        Write-Step 'Building the frontend (not in the repository)'
        Install-IfMissing 'node' 'OpenJS.NodeJS.LTS' 'Node.js'
        # Corepack otherwise stops to ask permission to fetch the pinned pnpm,
        # and waits forever for an answer nobody is there to give.
        $env:COREPACK_ENABLE_DOWNLOAD_PROMPT = '0'
        Push-Location (Join-Path $Root 'frontend')
        corepack pnpm install
        corepack pnpm run build
        Pop-Location
        Move-Item (Join-Path $Root 'frontend\out') $export
    }

    Write-Step 'Environment'
    $venv = Join-Path $Root '.venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $python)) {
        Write-Host "Creating $venv"
        py -3.13 -m venv $venv
    }
    else {
        Write-Host "Reusing $venv"
    }
    & $python -m pip install --quiet --upgrade pip
    # pywebview pulls in pythonnet, which is the whole Windows backend, and
    # window_chrome logs through loguru. No ML dependencies beyond that: the
    # window frame does not need them.
    & $python -m pip install --quiet pywebview loguru
    & $python -m pip show pywebview loguru | Select-String '^(Name|Version):'

    Write-Step 'Running the check'
    Write-Host 'A window opens, the probes run a few seconds later, and the window stays open afterwards.'
    Write-Host 'Close it when you are done with the manual checks it lists.'
    & $python $checkScript
}
catch {
    Write-Host ''
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Stop-Transcript | Out-Null
    Write-Host ''
    Write-Host "Everything above is saved in $LogFile" -ForegroundColor Green
}
