#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Rebuild, tag, and push the Cargia Docker image to Docker Hub.
    
.PARAMETER t
    The tag to use for the Docker image (e.g., "latest", "v1.0", "step1-test")
    
.EXAMPLE
    .\docker_rebuild.ps1 -t "latest"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$t
)

$ImageName = "cargia-trainer"
$Repository = "INSERT_REPO_HERE"
$FullImageName = "$Repository`:$t"

Write-Host "=== Cargia Docker Rebuild Script ==="
Write-Host "Tag: $t"
Write-Host "Repository: $Repository"
Write-Host "Full Image Name: $FullImageName"
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..."
docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop."
    exit 1
}
Write-Host "Docker is running"
Write-Host ""

# Build the image
Write-Host "Building Docker image..."
docker build -t $ImageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed"
    exit 1
}
Write-Host "Docker image built successfully"
Write-Host ""

# Tag the image
Write-Host "Tagging image..."
docker tag $ImageName $FullImageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Image tagging failed"
    exit 1
}
Write-Host "Image tagged successfully"
Write-Host ""

# Push the image
Write-Host "Pushing to Docker Hub..."
docker push $FullImageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Image push failed"
    exit 1
}
Write-Host "Image pushed successfully to Docker Hub"
Write-Host ""

# Success message
Write-Host "=== SUCCESS! ==="
Write-Host "Your Docker image has been built, tagged, and pushed successfully!"
Write-Host ""
Write-Host "On RunPod, you can now pull it with:"
Write-Host "  docker pull $FullImageName"
Write-Host ""
Write-Host "And run it with:"
Write-Host "  docker run --gpus all -v /path/to/weights:/weights $FullImageName"
Write-Host "" 