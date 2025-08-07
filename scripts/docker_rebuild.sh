#!/bin/bash
# Rebuild and restart Shagun Intelligence Trading System
echo "Rebuilding Shagun Intelligence Trading System..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d
