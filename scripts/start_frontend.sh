#!/bin/bash

# Shagun Intelligence Trading System - Frontend Startup Script
# This script starts the React-based trading dashboard

set -e

echo "🚀 Starting Shagun Intelligence Trading Dashboard..."

# Check if we're in the right directory
if [ ! -f "dashboard/package.json" ]; then
    echo "❌ Error: dashboard/package.json not found. Please run from project root."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Navigate to dashboard directory
cd dashboard

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dashboard dependencies..."
    npm install
else
    echo "✅ Dependencies already installed"
fi

# Check if backend is running
echo "🔍 Checking if backend is running..."
if curl -s http://127.0.0.1:8000/api/v1/health > /dev/null; then
    echo "✅ Backend is running at http://127.0.0.1:8000"
else
    echo "⚠️  Warning: Backend not detected at http://127.0.0.1:8000"
    echo "   Please start the backend first:"
    echo "   poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000"
    echo ""
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the development server
echo "🌐 Starting React development server..."
echo "📱 Dashboard will be available at: http://localhost:5173"
echo "🔗 Backend API: http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Start with environment variables for API connection
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
