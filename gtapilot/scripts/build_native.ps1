# Minimal build script for gtapilot/native using vcpkg manifest + CMake (VS)

[CmdletBinding()]
param(
  [switch]$Clean,
  [switch]$Clangd
)

$ErrorActionPreference = 'Stop'
Write-Host "=== gtapilot native build ==="

# Resolve roots
$ScriptPath = if ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
$GtapilotDir = Resolve-Path (Join-Path $ScriptPath "..")
$RepoRoot    = Split-Path $GtapilotDir -Parent
$NativeDir   = Join-Path $GtapilotDir "native"
$BuildDir    = Join-Path $RepoRoot "build\win-rel"
$BinDir      = Join-Path $RepoRoot "bin"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $BinDir   | Out-Null
if ($Clean) { Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null }

# Tool checks
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "CMake not found. Install CMake or use 'Developer PowerShell for VS'." }
if (-not (Get-Command git   -ErrorAction SilentlyContinue)) { throw "Git not found. Install Git for Windows." }

# vcpkg manifest mode: ensure vcpkg exists (use VCPKG_ROOT if set; else clone local copy)
function Ensure-VcpkgRoot {
  param([string]$ExistingRoot)
  if ($ExistingRoot) {
    $tc = Join-Path $ExistingRoot "scripts\buildsystems\vcpkg.cmake"
    if (Test-Path $tc) {
      return (Resolve-Path $ExistingRoot).Path
    }
  }
  $LocalVcpkg = Join-Path $RepoRoot ".vcpkg"
  $LocalToolchain = Join-Path $LocalVcpkg "scripts\buildsystems\vcpkg.cmake"
  if (!(Test-Path $LocalToolchain)) {
    if (!(Test-Path $LocalVcpkg)) {
      Write-Host "Cloning vcpkg -> $LocalVcpkg"
      git clone https://github.com/microsoft/vcpkg.git $LocalVcpkg | Out-Null
    }
    & "$LocalVcpkg\bootstrap-vcpkg.bat"
  }
  return (Resolve-Path $LocalVcpkg).Path
}

$Env:VCPKG_ROOT = Ensure-VcpkgRoot $Env:VCPKG_ROOT
# Sanitize any stray newlines or echoed content from bootstrap
$Env:VCPKG_ROOT = [string]$Env:VCPKG_ROOT
$Env:VCPKG_ROOT = ($Env:VCPKG_ROOT -split "(\r\n|\n)")[0]
$Toolchain = Join-Path $Env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake"
if (!(Test-Path $Toolchain)) { throw "Could not find vcpkg toolchain file at: $Toolchain" }
Write-Host "Using vcpkg: $Env:VCPKG_ROOT"

# If an old cache points to a missing/other toolchain, wipe the build dir to avoid CMake early failure
$Cache = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $Cache) {
  $cacheText = Get-Content $Cache -Raw
  if ($cacheText -match "CMAKE_TOOLCHAIN_FILE:STRING=(.+)") {
    $cachedPath = $Matches[1]
    $same = [System.IO.Path]::GetFullPath($cachedPath) -eq [System.IO.Path]::GetFullPath($Toolchain)
    if (-not $same -or -not (Test-Path $cachedPath)) {
      Write-Host "Stale CMake cache detected (toolchain mismatch). Cleaning build dir ..."
      Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
      New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    }
  }
}

# Configure with Visual Studio generator (works without Ninja)
Write-Host "Configuring CMake (VS 2022, x64) ..."
cmake -G "Visual Studio 17 2022" -A x64 -S "$NativeDir" -B "$BuildDir" -DCMAKE_TOOLCHAIN_FILE="$Toolchain" -DCMAKE_BUILD_TYPE=Release | Out-Host
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }

# Build Release
Write-Host "Building Release ..."
cmake --build "$BuildDir" --config Release --parallel | Out-Host
if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }

# Copy artifacts to bin
Write-Host "Copying artifacts -> $BinDir"
Get-ChildItem -Path (Join-Path $BuildDir "Release") -Filter *.exe -Recurse -ErrorAction SilentlyContinue | Copy-Item -Destination $BinDir -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $BuildDir "Release") -Filter *.dll -Recurse -ErrorAction SilentlyContinue | Copy-Item -Destination $BinDir -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $BuildDir "Release") -Filter *.pdb -Recurse -ErrorAction SilentlyContinue | Copy-Item -Destination $BinDir -Force -ErrorAction SilentlyContinue

Write-Host "Binaries are in: $BinDir"
Write-Host "=== Build complete ==="

# Optionally generate compile_commands.json for clangd (requires Ninja)
if ($Clangd) {
  if ($null -eq (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Warning "Clangd mode requested but Ninja not found. Install Ninja (choco install ninja) to generate compile_commands.json."
  } else {
    $ClangdDir = Join-Path $RepoRoot "build\\clangd"
    New-Item -ItemType Directory -Force -Path $ClangdDir | Out-Null
    Write-Host "Configuring clangd compile database (Ninja) ..."
    cmake -G "Ninja" -S "$NativeDir" -B "$ClangdDir" -DCMAKE_TOOLCHAIN_FILE="$Toolchain" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON | Out-Host
    if (Test-Path (Join-Path $ClangdDir "compile_commands.json")) {
      Copy-Item (Join-Path $ClangdDir "compile_commands.json") (Join-Path $RepoRoot "compile_commands.json") -Force
      Write-Host "Wrote compile_commands.json at repo root for clangd."
    } else {
      Write-Warning "Failed to create compile_commands.json in $ClangdDir"
    }
  }
}
