# Shagun Intelligence Disaster Recovery Runbook

## Table of Contents
1. [Overview](#overview)
2. [Contact Information](#contact-information)
3. [Disaster Scenarios](#disaster-scenarios)
4. [Recovery Procedures](#recovery-procedures)
5. [Validation Steps](#validation-steps)
6. [Post-Recovery Actions](#post-recovery-actions)

## Overview

This runbook provides step-by-step procedures for recovering Shagun Intelligence services in case of various disaster scenarios.

**RTO (Recovery Time Objective)**: 4 hours  
**RPO (Recovery Point Objective)**: 1 hour

## Contact Information

### Emergency Contacts
- **On-Call Engineer**: Check PagerDuty
- **Engineering Lead**: engineering-lead@shagunintelligence.com
- **DevOps Team**: devops@shagunintelligence.com
- **Security Team**: security@shagunintelligence.com

### External Contacts
- **AWS Support**: 1-800-xxx-xxxx (Enterprise Support)
- **Zerodha API Support**: api-support@zerodha.com

## Disaster Scenarios

### 1. Complete Region Failure

#### Detection
- All services in primary region (us-east-1) are unreachable
- AWS status page shows region-wide issues

#### Recovery Steps

1. **Activate DR Region (us-west-2)**
```bash
# Switch DNS to DR region
./scripts/dr-failover.sh activate-dr us-west-2

# Verify DR endpoints
curl https://dr.shagunintelligence.com/api/v1/health
```

2. **Restore Latest Data**
```bash
# Find latest cross-region backup
aws s3 ls s3://shagunintelligence-dr-backups/ --recursive | grep backup | tail -1

# Restore to DR database
./scripts/restore-dr.sh <backup-name>
```

3. **Update Configuration**
```bash
# Update Kubernetes configs for DR region
kubectl apply -k k8s/overlays/dr-west/

# Verify all pods are running
kubectl get pods -n shagunintelligence
```

### 2. Database Corruption/Loss

#### Detection
- Database queries failing with corruption errors
- Unexpected data inconsistencies
- Backup validation failures

#### Recovery Steps

1. **Stop Application Traffic**
```bash
# Scale down application
kubectl scale deployment shagunintelligence-app --replicas=0 -n shagunintelligence

# Put maintenance page
kubectl apply -f k8s/maintenance/maintenance-page.yaml
```

2. **Restore Database**
```bash
# List available backups
aws s3 ls s3://shagunintelligence-backups/ | grep database

# Restore specific backup
./scripts/restore-database.sh <backup-timestamp>

# Verify database integrity
psql -h postgres-service -U shagunintelligence -c "SELECT COUNT(*) FROM trades;"
```

3. **Validate Data Consistency**
```bash
# Run consistency checks
python scripts/validate-database.py

# Check for missing trades
python scripts/audit-trades.py --from "2024-01-01" --to "now"
```

### 3. Kubernetes Cluster Failure

#### Detection
- Unable to connect to Kubernetes API
- Multiple node failures
- Control plane unresponsive

#### Recovery Steps

1. **Activate Backup Cluster**
```bash
# Switch context to backup cluster
kubectl config use-context shagunintelligence-backup

# Apply latest configurations
kubectl apply -k k8s/base/
kubectl apply -k k8s/overlays/production/
```

2. **Restore Persistent Volumes**
```bash
# Restore from Velero backup
velero restore create --from-backup daily-backup-latest

# Monitor restore progress
velero restore describe daily-backup-latest --details
```

### 4. Security Breach

#### Detection
- Unusual API activity patterns
- Unauthorized access attempts
- Compromised credentials alert

#### Recovery Steps

1. **Immediate Containment**
```bash
# Disable all API keys
python scripts/disable-all-keys.py

# Rotate all secrets
./scripts/rotate-secrets.sh --all

# Enable emergency mode (read-only)
kubectl set env deployment/shagunintelligence-app EMERGENCY_MODE=true -n shagunintelligence
```

2. **Investigation**
```bash
# Export audit logs
kubectl logs -n shagunintelligence -l app=shagunintelligence-app --since=24h > audit.log

# Check for unauthorized trades
python scripts/audit-security.py --check-unauthorized
```

3. **Recovery**
```bash
# Generate new API keys
python scripts/generate-new-keys.py

# Update Kubernetes secrets
kubectl create secret generic shagunintelligence-secrets --from-file=secrets.yaml -n shagunintelligence

# Resume normal operations
kubectl set env deployment/shagunintelligence-app EMERGENCY_MODE=false -n shagunintelligence
```

## Recovery Procedures

### Standard Recovery Process

1. **Assessment Phase** (15 minutes)
   - Identify the scope of the disaster
   - Determine affected components
   - Estimate recovery time

2. **Communication** (5 minutes)
   - Notify stakeholders via Slack/Email
   - Update status page
   - Create incident in PagerDuty

3. **Recovery Execution** (Variable)
   - Follow specific scenario steps
   - Document all actions taken
   - Coordinate with team members

4. **Validation** (30 minutes)
   - Run automated test suite
   - Verify critical functionality
   - Check data consistency

5. **Declaration** (10 minutes)
   - Announce recovery completion
   - Update status page
   - Schedule post-mortem

### Database Recovery from Backup

```bash
#!/bin/bash
# restore-database.sh

BACKUP_NAME=$1
TEMP_DB="shagunintelligence_restore_temp"

# Create temporary database
psql -h postgres-service -U postgres -c "CREATE DATABASE $TEMP_DB;"

# Restore to temporary database
gunzip -c backups/${BACKUP_NAME}/database.sql.gz | \
  psql -h postgres-service -U postgres -d $TEMP_DB

# Validate restore
RECORD_COUNT=$(psql -h postgres-service -U postgres -d $TEMP_DB -t -c "SELECT COUNT(*) FROM trades;")
echo "Restored $RECORD_COUNT trade records"

# Swap databases
psql -h postgres-service -U postgres <<EOF
ALTER DATABASE shagunintelligence RENAME TO shagunintelligence_old;
ALTER DATABASE $TEMP_DB RENAME TO shagunintelligence;
EOF

echo "Database restore complete"
```

### Application Rollback

```bash
#!/bin/bash
# rollback-deployment.sh

# Get previous revision
PREV_REVISION=$(kubectl rollout history deployment/shagunintelligence-app -n shagunintelligence | tail -2 | head -1 | awk '{print $1}')

# Rollback to previous revision
kubectl rollout undo deployment/shagunintelligence-app --to-revision=$PREV_REVISION -n shagunintelligence

# Monitor rollback
kubectl rollout status deployment/shagunintelligence-app -n shagunintelligence

# Verify health
curl https://shagunintelligence.com/api/v1/health
```

## Validation Steps

### 1. Health Checks
```bash
# API Health
curl https://shagunintelligence.com/api/v1/health

# Database connectivity
psql -h postgres-service -U shagunintelligence -c "SELECT 1;"

# Redis connectivity
redis-cli -h redis-service ping

# WebSocket connectivity
wscat -c wss://shagunintelligence.com/ws
```

### 2. Functional Tests
```bash
# Run smoke tests
pytest tests/smoke/ -v

# Test trading functionality (dry run)
python scripts/test-trading.py --dry-run

# Verify AI agents
python scripts/test-agents.py
```

### 3. Data Validation
```bash
# Check data integrity
python scripts/validate-data.py --full

# Compare with backup
python scripts/compare-backup.py --latest

# Audit recent trades
python scripts/audit-trades.py --hours 24
```

## Post-Recovery Actions

1. **Documentation**
   - Create incident report
   - Update runbook with lessons learned
   - Document any deviations from procedure

2. **Communication**
   - Send all-clear to stakeholders
   - Update status page
   - Notify customers if needed

3. **Follow-up**
   - Schedule post-mortem (within 48 hours)
   - Create action items
   - Update monitoring/alerts

4. **Testing**
   - Schedule DR drill
   - Update recovery procedures
   - Train team on new procedures

## Appendix

### Useful Commands

```bash
# Get all logs for debugging
kubectl logs -n shagunintelligence -l app=shagunintelligence-app --tail=1000

# Check resource usage
kubectl top pods -n shagunintelligence

# Database connection string
postgresql://shagunintelligence:password@postgres-service:5432/shagunintelligence

# Redis connection
redis-cli -h redis-service -a password

# Port forwarding for debugging
kubectl port-forward svc/postgres-service 5432:5432 -n shagunintelligence
```

### Recovery Metrics

Track these metrics during recovery:
- Time to detection
- Time to recovery initiation
- Time to service restoration
- Data loss (if any)
- Number of affected users
- Financial impact

### DR Testing Schedule

- **Monthly**: Backup restoration test
- **Quarterly**: Partial failover test
- **Annually**: Full DR simulation