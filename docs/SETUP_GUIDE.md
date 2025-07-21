# AlgoHive Setup Guide

This guide provides detailed instructions for setting up AlgoHive in various environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Development Setup](#development-setup)
4. [Production Setup](#production-setup)
5. [Configuration Details](#configuration-details)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows 10+ with WSL2

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 22.04 LTS

## Prerequisites Installation

### 1. Install Python 3.11+

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### macOS
```bash
brew install python@3.11
```

#### Windows
Download from [python.org](https://www.python.org/downloads/) or use:
```powershell
winget install Python.Python.3.11
```

### 2. Install Docker

#### Ubuntu
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### macOS
Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)

#### Windows
Download Docker Desktop with WSL2 backend from [docker.com](https://www.docker.com/products/docker-desktop/)

### 3. Install PostgreSQL

#### Using Docker (Recommended)
```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=algohive \
  -e POSTGRES_USER=algohive \
  -e POSTGRES_DB=algohive \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15-alpine
```

#### Native Installation
```bash
# Ubuntu
sudo apt install postgresql-15 postgresql-contrib-15

# macOS
brew install postgresql@15
brew services start postgresql@15
```

### 4. Install Redis

#### Using Docker (Recommended)
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine redis-server --appendonly yes
```

#### Native Installation
```bash
# Ubuntu
sudo apt install redis-server

# macOS
brew install redis
brew services start redis
```

### 5. Install Node.js 18+ (for Dashboard)

```bash
# Using NodeSource (Ubuntu)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node@18

# Verify installation
node --version
npm --version
```

### 6. Install TA-Lib

TA-Lib is required for technical analysis calculations.

#### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential wget

# Download and install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update library path
echo "export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

#### macOS
```bash
brew install ta-lib
```

#### Windows (WSL2)
Follow the Ubuntu instructions inside WSL2.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/algohive/algohive.git
cd algohive
```

### 2. Create Python Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development requirements
pip install -r requirements-dev.txt
```

### 4. Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your favorite editor
nano .env
```

Required environment variables:
```env
# Application Settings
APP_ENV=development
SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://algohive:algohive@localhost:5432/algohive

# Redis
REDIS_URL=redis://localhost:6379/0

# Zerodha Kite API (Get from https://kite.trade)
KITE_API_KEY=your-api-key
KITE_API_SECRET=your-api-secret
KITE_ACCESS_TOKEN=your-access-token  # Generated after login

# AI Services (Optional but recommended)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Trading Settings
MAX_POSITION_SIZE=100000
MAX_DAILY_TRADES=10
RISK_PER_TRADE=2.0
STOP_LOSS_PERCENT=2.0
```

### 5. Database Setup

```bash
# Create database
createdb algohive

# Run migrations
alembic upgrade head

# Create initial user (optional)
python scripts/create_user.py --username admin --password admin123
```

### 6. Start Development Servers

#### Option 1: Using Docker Compose
```bash
docker-compose -f docker-compose.dev.yml up
```

#### Option 2: Manual Start
```bash
# Terminal 1: Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Celery Worker (if using)
celery -A app.celery worker --loglevel=info

# Terminal 3: Start Dashboard
cd dashboard
npm install
npm start
```

### 7. Verify Installation

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check database connection
python scripts/check_db.py

# Run tests
pytest tests/unit/test_health.py -v
```

## Production Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
  build-essential \
  curl \
  git \
  nginx \
  certbot \
  python3-certbot-nginx \
  supervisor
```

### 2. Clone and Configure

```bash
# Clone repository
cd /opt
sudo git clone https://github.com/algohive/algohive.git
sudo chown -R $USER:$USER algohive
cd algohive

# Create production environment
cp .env.production .env
# Edit .env with production values
```

### 3. Docker Production Setup

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f
```

### 4. Nginx Configuration

Create `/etc/nginx/sites-available/algohive`:

```nginx
server {
    listen 80;
    server_name algohive.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/algohive /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 5. SSL Certificate

```bash
sudo certbot --nginx -d algohive.yourdomain.com
```

### 6. Systemd Service

Create `/etc/systemd/system/algohive.service`:

```ini
[Unit]
Description=AlgoHive Trading Platform
After=network.target

[Service]
Type=forking
User=algohive
WorkingDirectory=/opt/algohive
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable algohive
sudo systemctl start algohive
```

## Configuration Details

### Zerodha Kite Setup

1. **Create Kite Connect App**
   - Go to [https://kite.trade](https://kite.trade)
   - Create a new app
   - Note down API Key and Secret

2. **Generate Access Token**
   ```python
   from services.kite import KiteService
   
   kite = KiteService()
   login_url = kite.get_login_url()
   print(f"Visit: {login_url}")
   
   # After login, get request_token from URL
   access_token = kite.generate_session(request_token)
   print(f"Access Token: {access_token}")
   ```

3. **Configure Webhooks** (Optional)
   - Set postback URL: `https://yourdomain.com/api/v1/webhooks/kite`

### Database Configuration

#### Connection Pool Settings
```env
# Add to .env for production
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=1800
```

#### Backup Configuration
```bash
# Create backup script
cat > /opt/algohive/scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/algohive"
mkdir -p $BACKUP_DIR
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/algohive_$(date +%Y%m%d_%H%M%S).sql.gz
find $BACKUP_DIR -type f -mtime +7 -delete
EOF

chmod +x /opt/algohive/scripts/backup.sh

# Add to crontab
0 2 * * * /opt/algohive/scripts/backup.sh
```

### Redis Configuration

For production, configure Redis persistence:

```bash
# /etc/redis/redis.conf
appendonly yes
appendfsync everysec
maxmemory 2gb
maxmemory-policy allkeys-lru
```

## Troubleshooting

### Common Issues

#### 1. TA-Lib Import Error
```
ImportError: libta_lib.so.0: cannot open shared object file
```

**Solution:**
```bash
sudo ldconfig
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### 2. Database Connection Error
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify connection string in .env
- Check firewall rules

#### 3. Redis Connection Error
```
redis.exceptions.ConnectionError: Error -2 connecting to redis:6379
```

**Solution:**
- Check Redis is running: `redis-cli ping`
- Verify Redis URL in .env
- Check if Redis is bound to localhost only

#### 4. Docker Permission Error
```
permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 5. Port Already in Use
```
Error: bind: address already in use
```

**Solution:**
```bash
# Find process using port
sudo lsof -i :8000
# Kill process
sudo kill -9 <PID>
```

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
DEBUG=True
```

Check logs:
```bash
# Application logs
tail -f logs/algohive.log

# Docker logs
docker-compose logs -f app

# System logs
journalctl -u algohive -f
```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- Create indexes
   CREATE INDEX idx_trades_symbol ON trades(symbol);
   CREATE INDEX idx_trades_timestamp ON trades(timestamp);
   ```

2. **Redis Optimization**
   ```bash
   # Increase max clients
   redis-cli CONFIG SET maxclients 10000
   ```

3. **Application Optimization**
   ```env
   # Increase workers
   WORKERS=4
   # Enable uvloop
   USE_UVLOOP=true
   ```

## Next Steps

1. [API Documentation](API_DOCUMENTATION.md)
2. [Agent Configuration](AGENT_ARCHITECTURE.md)
3. [Trading Strategies](TRADING_STRATEGIES.md)
4. [Security Best Practices](SECURITY.md)