# S3 Backup Storage Setup Guide

This guide explains how to set up AWS S3 for storing R3MES database backups.

## Prerequisites

- AWS account
- AWS CLI installed and configured
- S3 bucket created

## Setup Steps

### 1. Create S3 Bucket

```bash
aws s3 mb s3://r3mes-backups --region us-east-1
```

### 2. Configure Lifecycle Policy

Create a lifecycle policy to automatically move old backups to Glacier:

```json
{
  "Rules": [
    {
      "Id": "BackupLifecycle",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    }
  ]
}
```

Apply the policy:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket r3mes-backups \
  --lifecycle-configuration file://lifecycle-policy.json
```

### 3. Configure IAM Policy

Create an IAM policy for backup uploads:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::r3mes-backups",
        "arn:aws:s3:::r3mes-backups/*"
      ]
    }
  ]
}
```

### 4. Configure Environment Variables

Set the following environment variables:

```bash
export S3_BUCKET=r3mes-backups
export AWS_REGION=us-east-1
```

### 5. Test Upload

```bash
./scripts/backup_s3_upload.sh /backups/r3mes_20241224_120000.sql.gz
```

## Automated Backup with S3

The backup script automatically uploads to S3 if `S3_BUCKET` is set:

```bash
export S3_BUCKET=r3mes-backups
./scripts/backup_database.sh
```

## Restore from S3

```bash
# Download from S3
aws s3 cp s3://r3mes-backups/database-backups/r3mes_20241224_120000.sql.gz /backups/

# Restore
./scripts/restore_database.sh /backups/r3mes_20241224_120000.sql.gz
```

## Cost Optimization

- Use lifecycle policies to move old backups to Glacier
- Enable S3 Intelligent-Tiering for automatic cost optimization
- Set expiration policies to delete very old backups

