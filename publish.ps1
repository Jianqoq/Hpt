$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
$certPath = Join-Path $cargoHome "cacert.pem"
$env:CARGO_HTTP_CAINFO = $certPath

if (-not (Test-Path $certPath)) {
    Write-Host "Downloading SSL certificate..."
    New-Item -ItemType Directory -Force -Path $cargoHome | Out-Null
    Invoke-WebRequest -Uri "https://curl.se/ca/cacert.pem" -OutFile $certPath
}

$ErrorActionPreference = "Stop"

$packages = @(
    "hpt-macros",
    "hpt-types",
    "hpt-common",
    "hpt-traits",
    "hpt-iterator",
    "hpt-display",
    "hpt-allocator",
    "hpt-dataloader",
    "hpt-cudakernels",
    "hpt"
)

$initialLocation = Get-Location

try {
    foreach ($package in $packages) {
        Write-Host "`n========================================"
        Write-Host "Processing package: $package"
        Write-Host "========================================`n"
        
        Push-Location $package
        
        try {
            Write-Host "Running dry run check..."
            cargo publish --dry-run
            "Dry run successful"
            Write-Host "Publishing $package..."
            cargo publish
            Write-Host "$package published successfully!"
        }
        catch {
            Write-Host "Error processing $package"
            Write-Host $_.Exception.Message
            $continue = Read-Host "Continue with next package? (Y/N)"
            if ($continue -ne 'Y' -and $continue -ne 'y') {
                throw "Operation cancelled by user"
            }
        }
        finally {
            Pop-Location
        }
    }
}
catch {
    Write-Host "Error occurred:"
    Write-Host $_.Exception.Message
}
finally {
    Set-Location $initialLocation
}

Write-Host "`nAll packages processed!"