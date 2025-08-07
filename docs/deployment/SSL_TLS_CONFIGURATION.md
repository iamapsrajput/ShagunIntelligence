# SSL/TLS Configuration Guide

## Overview

This guide provides comprehensive instructions for setting up SSL/TLS certificates for the Shagun Intelligence Trading Platform in production environments.

## 🔒 SSL/TLS Setup Options

### Option 1: Let's Encrypt (Recommended for Production)

#### Prerequisites

- Domain name pointing to your server
- Port 80 and 443 accessible
- Root access to the server

#### Automated Setup Script

```bash
# Run the SSL setup script
chmod +x scripts/ssl_setup.sh
sudo ./scripts/ssl_setup.sh your-domain.com
```

#### Manual Let's Encrypt Setup

```bash
# Install Certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Verify auto-renewal
sudo certbot renew --dry-run
```

### Option 2: Self-Signed Certificates (Development/Testing)

```bash
# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/shagun-selfsigned.key \
    -out /etc/ssl/certs/shagun-selfsigned.crt

# Create Diffie-Hellman group
sudo openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
```

## 🔧 Nginx SSL Configuration

### Production SSL Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/your-domain.com/chain.pem;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # Application proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🚀 Docker SSL Configuration

### Docker Compose with SSL

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - trading-app

  trading-app:
    build: .
    environment:
      - ENVIRONMENT=production
      - SSL_ENABLED=true
    expose:
      - "8000"
```

## 🔄 Certificate Management

### Automatic Renewal

```bash
# Add to crontab for automatic renewal
0 12 * * * /usr/bin/certbot renew --quiet
```

### Manual Renewal

```bash
# Renew certificates manually
sudo certbot renew

# Reload nginx after renewal
sudo systemctl reload nginx
```

### Certificate Monitoring

```bash
# Check certificate expiration
sudo certbot certificates

# Test certificate validity
openssl x509 -in /etc/letsencrypt/live/your-domain.com/cert.pem -text -noout
```

## 🛡️ Security Best Practices

### SSL/TLS Security Checklist

- [ ] Use TLS 1.2 or higher
- [ ] Disable weak ciphers
- [ ] Enable HSTS
- [ ] Implement OCSP stapling
- [ ] Use strong DH parameters
- [ ] Regular certificate renewal
- [ ] Monitor certificate expiration

### Security Headers

```nginx
# Comprehensive security headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
```

## 🧪 Testing SSL Configuration

### SSL Labs Test

```bash
# Test your SSL configuration
curl -s "https://api.ssllabs.com/api/v3/analyze?host=your-domain.com"
```

### Local SSL Testing

```bash
# Test SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check certificate chain
openssl s_client -connect your-domain.com:443 -showcerts
```

## 🔧 Troubleshooting

### Common SSL Issues

1. **Certificate Not Found**

   ```bash
   # Check certificate files exist
   ls -la /etc/letsencrypt/live/your-domain.com/
   ```

2. **Permission Issues**

   ```bash
   # Fix certificate permissions
   sudo chmod 644 /etc/letsencrypt/live/your-domain.com/fullchain.pem
   sudo chmod 600 /etc/letsencrypt/live/your-domain.com/privkey.pem
   ```

3. **Nginx Configuration Errors**

   ```bash
   # Test nginx configuration
   sudo nginx -t

   # Reload nginx
   sudo systemctl reload nginx
   ```

### SSL Certificate Renewal Issues

```bash
# Debug renewal issues
sudo certbot renew --dry-run --verbose

# Force renewal
sudo certbot renew --force-renewal
```

## 📊 Monitoring and Alerts

### Certificate Expiration Monitoring

```bash
# Script to check certificate expiration
#!/bin/bash
DOMAIN="your-domain.com"
EXPIRY_DATE=$(openssl s_client -connect $DOMAIN:443 -servername $DOMAIN 2>/dev/null | openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_UNTIL_EXPIRY=$(( ($EXPIRY_EPOCH - $CURRENT_EPOCH) / 86400 ))

if [ $DAYS_UNTIL_EXPIRY -lt 30 ]; then
    echo "WARNING: SSL certificate expires in $DAYS_UNTIL_EXPIRY days"
fi
```

---

**This SSL/TLS configuration ensures secure, encrypted communication for the Shagun Intelligence Trading Platform with industry-standard security practices.**
