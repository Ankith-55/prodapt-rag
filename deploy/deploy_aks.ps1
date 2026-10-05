# Build, push and deploy to AKS.   Usage (from the repo root, after `docker compose build`):
#   .\deploy\deploy_aks.ps1 -Acr <registry-name>
# Prerequisites: az login, an ACR and an AKS cluster (see docs/production_scale.md), Docker Desktop, kubectl.
# Safe to re-run: every step is idempotent. Native command failures stop the script (PowerShell does not do that by default).
param(
    [Parameter(Mandatory = $true)][string]$Acr,
    [string]$ResourceGroup = "rg-ticketrag",
    [string]$Cluster = "aks-ticketrag",
    [string]$Tag = "0.1.0"
)
$ErrorActionPreference = "Stop"
$server = "$Acr.azurecr.io"

# Runs a native command, retries on failure (e.g. a dropped connection to the cluster API), throws if it never succeeds.
function Run([scriptblock]$Block, [string]$What, [int]$Tries = 4) {
    for ($i = 1; $i -le $Tries; $i++) {
        & $Block
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "  '$What' failed (attempt $i of $Tries)" -ForegroundColor Yellow
        if ($i -lt $Tries) { Start-Sleep -Seconds (5 * $i) }
    }
    throw "'$What' failed after $Tries attempts"
}

Write-Host "1/7 log in to the registry"
az acr login --name $Acr
if ($LASTEXITCODE -ne 0) {
    # `az acr login` runs its own connectivity probe, which some networks block even when Docker itself can reach the
    # registry. Fall back to the registry admin user (demo only: disable it afterwards, see docs/production_scale.md).
    Write-Host "  az acr login failed; falling back to registry admin credentials" -ForegroundColor Yellow
    az acr update --name $Acr --admin-enabled true | Out-Null
    $user = az acr credential show --name $Acr --query username -o tsv
    $pass = az acr credential show --name $Acr --query "passwords[0].value" -o tsv
    $pass | docker login $server --username $user --password-stdin
    if ($LASTEXITCODE -ne 0) { throw "could not log in to $server" }
}

Write-Host "2/7 push the application image"
Run { docker tag ticketrag:local "$server/ticketrag:$Tag" } "docker tag"
Run { docker push "$server/ticketrag:$Tag" } "push application image"

Write-Host "3/7 stage the ticket state and build the state image"
$stage = "deploy/aks-build/state"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory $stage | Out-Null
$src = if (Test-Path data/processed/patterns.parquet) { "data/processed" } else { "artifacts/state" }
Copy-Item "$src/patterns.parquet", "$src/resolutions.parquet", "$src/tickets.parquet" $stage
Copy-Item "$src/index" "$stage/index" -Recurse
Run { docker build -f deploy/aks-build/Dockerfile --build-arg BASE="$server/ticketrag:$Tag" -t "$server/ticketrag-demo:$Tag" deploy/aks-build } "build state image" 1
Run { docker push "$server/ticketrag-demo:$Tag" } "push state image"

Write-Host "4/7 connect kubectl to the cluster and let it pull from the registry"
Run { az aks get-credentials --resource-group $ResourceGroup --name $Cluster --overwrite-existing } "get credentials"
Run { az aks update --resource-group $ResourceGroup --name $Cluster --attach-acr $Acr | Out-Null } "attach registry"

Write-Host "5/7 namespace, config and secret"
Run { kubectl apply -f deploy/k8s/namespace.yaml } "apply namespace"
Run { kubectl apply -f deploy/k8s/configmap.yaml } "apply configmap"
$line = (Get-Content .env | Where-Object { $_ -match '^\s*OPENAI_API_KEY\s*=' } | Select-Object -First 1)
if (-not $line) { throw "OPENAI_API_KEY not found in .env" }
$openai = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
$access = [guid]::NewGuid().ToString("N")
Run {
    kubectl -n ticketrag create secret generic ticketrag-secrets `
        --from-literal=OPENAI_API_KEY=$openai --from-literal=TICKETRAG_API_KEY=$access `
        --dry-run=client -o yaml | kubectl apply -f -
} "apply secret"

Write-Host "6/7 deployment, service and autoscaler"
Run { (Get-Content deploy/k8s/deployment.yaml -Raw).Replace("REPLACE_ACR", $Acr).Replace("REPLACE_TAG", $Tag) | kubectl apply -f - } "apply deployment"
Run { kubectl apply -f deploy/k8s/service.yaml } "apply service"
Run { kubectl apply -f deploy/k8s/hpa.yaml } "apply autoscaler"
# The first pod may already be stuck in ImagePullBackOff from an earlier attempt: restart it now that the image exists.
kubectl -n ticketrag rollout restart deployment/ticketrag | Out-Null

Write-Host "7/7 waiting for the rollout"
kubectl -n ticketrag rollout status deployment/ticketrag --timeout=420s
$rolled = ($LASTEXITCODE -eq 0)
kubectl -n ticketrag get pods

Write-Host ""
if ($rolled) {
    Write-Host "Deployed. Open the UI with:   kubectl -n ticketrag port-forward svc/ticketrag 8000:80"
    Write-Host "then http://localhost:8000/   Access key for the UI field / X-API-Key header: $access"
} else {
    Write-Host "Rollout did NOT complete. Diagnose with:" -ForegroundColor Red
    Write-Host "  kubectl -n ticketrag describe pod -l app.kubernetes.io/name=ticketrag"
    Write-Host "  kubectl -n ticketrag logs deploy/ticketrag"
    exit 1
}
