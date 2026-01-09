# Kubernetes Deployment Manifests

This directory contains Kubernetes manifests for deploying R3MES to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured to access the cluster
- Ingress controller (nginx-ingress recommended)
- cert-manager (for SSL certificates, optional)

## Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2. Create ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml
```

### 3. Create Secrets

**IMPORTANT**: First create secrets.yaml from the template:

```bash
cp k8s/secrets.yaml.template k8s/secrets.yaml
# Edit secrets.yaml and fill in base64 encoded values
kubectl apply -f k8s/secrets.yaml
```

To generate base64 encoded values:
```bash
echo -n "your-secret-value" | base64
```

### 4. Deploy PostgreSQL

```bash
kubectl apply -f k8s/postgres/
```

### 5. Deploy Redis

```bash
kubectl apply -f k8s/redis/
```

### 6. Deploy Backend

```bash
kubectl apply -f k8s/backend/
```

### 7. Deploy Frontend

```bash
kubectl apply -f k8s/frontend/
```

### 8. Deploy Ingress

```bash
kubectl apply -f k8s/ingress.yaml
```

## Health Checks

The deployments include liveness and readiness probes:

- **Backend**: 
  - Liveness: `/health`
  - Readiness: `/health/database`
- **Frontend**: 
  - Liveness: `/`
  - Readiness: `/`
- **PostgreSQL**: `pg_isready`
- **Redis**: `redis-cli ping`

## Scaling

To scale services:

```bash
# Scale backend
kubectl scale deployment r3mes-backend -n r3mes --replicas=3

# Scale frontend
kubectl scale deployment r3mes-frontend -n r3mes --replicas=3
```

## Monitoring

Deploy Prometheus and Grafana for monitoring:

```bash
# Apply monitoring stack (if available)
kubectl apply -f ../monitoring/k8s/
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n r3mes
```

### View Logs

```bash
# Backend logs
kubectl logs -f deployment/r3mes-backend -n r3mes

# Frontend logs
kubectl logs -f deployment/r3mes-frontend -n r3mes
```

### Describe Pod

```bash
kubectl describe pod <pod-name> -n r3mes
```

## Notes

- These manifests are production-ready templates
- Adjust resource limits based on your cluster capacity
- Update image tags to specific versions in production
- Configure persistent volumes for PostgreSQL and Redis
- Set up backup strategies for StatefulSets
- Configure network policies for security

