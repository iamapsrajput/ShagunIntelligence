# Sealed Secrets for Shagun Intelligence

This directory contains sealed secrets for secure secret management in Kubernetes.

## Setup

1. Install sealed-secrets controller:

```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml
```

2. Install kubeseal CLI:

```bash
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/kubeseal-0.18.0-linux-amd64.tar.gz
tar -xvzf kubeseal-0.18.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

## Creating Sealed Secrets

1. Create a regular secret:

```bash
kubectl create secret generic shagunintelligence-secrets \
  --from-literal=DB_PASSWORD=your-password \
  --from-literal=SECRET_KEY=your-secret-key \
  --dry-run=client -o yaml > secret.yaml
```

2. Seal the secret:

```bash
kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
```

3. Apply the sealed secret:

```bash
kubectl apply -f sealed-secret.yaml
```

## Environment-specific Secrets

- `sealed-secrets-staging.yaml` - Staging environment secrets
- `sealed-secrets-production.yaml` - Production environment secrets

## Important Notes

- Never commit unsealed secrets to git
- The private key is stored only in the sealed-secrets controller
- Sealed secrets can only be decrypted in the namespace they were created for
- Rotate secrets regularly using the rotation scripts
