[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [switch]$SkipVS
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)] [string]$Executable,
        [Parameter(Mandatory = $true)] [string[]]$Arguments
    )

    Write-Host "`n> $Executable $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command '$Executable' failed with exit code $LASTEXITCODE"
    }
}

function Initialize-VSEnvironment {
    if ($env:VSCMD_VER) {
        Write-Host "MSVC environment already configured (VSCMD_VER=$($env:VSCMD_VER))." -ForegroundColor DarkYellow
        return
    }

    $vswhereBase = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio'
    $vswherePath = Join-Path $vswhereBase 'Installer/vswhere.exe'
    if (-not (Test-Path $vswherePath)) {
        throw "vswhere.exe was not found. Install Visual Studio Build Tools 2022 and retry."
    }

    $vsInstall = & $vswherePath -latest -requires Microsoft.Component.MSBuild -property installationPath
    if (-not $vsInstall) {
        throw "Unable to locate a Visual Studio installation with C++ Build Tools."
    }

    $vsDevCmd = Join-Path $vsInstall 'Common7/Tools/VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat was not found at the expected path: $vsDevCmd"
    }

    Write-Host "Initializing MSVC environment via $vsDevCmd" -ForegroundColor DarkYellow
    $envDump = & cmd.exe /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 -no_logo && set"
    foreach ($line in $envDump) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            $name = $Matches['name']
            $value = $Matches['value']
            Set-Item -Path Env:$name -Value $value
        }
    }
    Write-Host "MSVC environment configured." -ForegroundColor DarkYellow
}

function Publish-Binaries {
    param(
        [Parameter(Mandatory = $true)] [string]$PresetName,
        [Parameter(Mandatory = $true)] [string]$Configuration,
        [Parameter(Mandatory = $true)] [string]$Destination
    )

    $sourceDir = Join-Path $repoRoot "build/$PresetName/bin/$Configuration"
    if (-not (Test-Path $sourceDir)) {
        Write-Host "No binaries to publish for preset '$PresetName' (missing $sourceDir)." -ForegroundColor DarkYellow
        return
    }

    if (-not (Test-Path $Destination)) {
        New-Item -Path $Destination -ItemType Directory | Out-Null
    }

    $exeSource = Join-Path $sourceDir 'DisplayCaptureDX11.exe'
    if (Test-Path $exeSource) {
        $exeTarget = Join-Path $Destination 'DisplayCaptureDX11.exe'
        Copy-Item -Path $exeSource -Destination $exeTarget -Force
        Write-Host "Copied $exeSource -> $exeTarget" -ForegroundColor Green
    }
    
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake was not found in PATH. Please install CMake and retry."
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "The 'ninja' build tool is required but was not found in PATH. Install Ninja before running this script."
}

Initialize-VSEnvironment

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '.')
Write-Host "Repository root: $repoRoot" -ForegroundColor DarkCyan

$pythonExe = $null
$venvPythonCandidates = @(
    (Join-Path $repoRoot '.venv/Scripts/python.exe')
    (Join-Path $repoRoot '.venv/bin/python')
)
foreach ($candidate in $venvPythonCandidates) {
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}
if (-not $pythonExe) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonExe = $pythonCmd.Source
    }
}

Push-Location $repoRoot
try {
    $ninjaPreset = 'ninja-multi'
    $vsPreset = 'vs2022'
    $binDir = Join-Path $repoRoot 'bin'

    Invoke-LoggedCommand -Executable 'cmake' -Arguments @('--preset', $ninjaPreset)

    $ninjaBuildPreset = "ninja-$Configuration"
    Invoke-LoggedCommand -Executable 'cmake' -Arguments @('--build', '--preset', $ninjaBuildPreset)

    $compileDb = Join-Path $repoRoot "build/$ninjaPreset/compile_commands.json"
    if (-not (Test-Path $compileDb)) {
        throw "Expected compile database not found at '$compileDb'."
    }

    $rootCompileDb = Join-Path $repoRoot 'compile_commands.json'
    $sanitizer = Join-Path $repoRoot 'tools/sanitize_compile_commands.py'
    if ((Test-Path $sanitizer) -and $pythonExe) {
        Invoke-LoggedCommand -Executable $pythonExe -Arguments @($sanitizer, $compileDb, '-o', $rootCompileDb)
    } else {
        Copy-Item -Path $compileDb -Destination $rootCompileDb -Force
        Write-Host "Sanitizer unavailable; raw compile_commands copied." -ForegroundColor DarkYellow
    }
    Write-Host "compile_commands.json updated at $rootCompileDb" -ForegroundColor Green

    Publish-Binaries -PresetName $ninjaPreset -Configuration $Configuration -Destination $binDir

    if (-not $SkipVS.IsPresent) {
        Invoke-LoggedCommand -Executable 'cmake' -Arguments @('--preset', $vsPreset)

        $vsBuildPreset = "${vsPreset}-$Configuration"
        Invoke-LoggedCommand -Executable 'cmake' -Arguments @('--build', '--preset', $vsBuildPreset)

        Publish-Binaries -PresetName $vsPreset -Configuration $Configuration -Destination $binDir
    }

    Write-Host "Native build completed successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
