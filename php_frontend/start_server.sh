#!/bin/bash

# BibleBot PHP Frontend Server Startup Script

echo "🚀 Starting BibleBot PHP Frontend Server..."
echo ""

# Check if PHP is installed
if ! command -v php &> /dev/null; then
    echo "❌ Error: PHP is not installed or not in PATH"
    echo "Please install PHP 7.4 or higher"
    exit 1
fi

# Check PHP version
PHP_VERSION=$(php -r "echo PHP_VERSION;")
echo "✅ PHP version: $PHP_VERSION"

# Check if cURL extension is enabled
if ! php -m | grep -q curl; then
    echo "❌ Error: cURL extension is not enabled"
    echo "Please enable the cURL extension in your PHP configuration"
    exit 1
fi
echo "✅ cURL extension: Enabled"

# Check if config.php exists
if [ ! -f "config.php" ]; then
    echo "❌ Error: config.php not found"
    echo "Please ensure you're running this script from the php_frontend directory"
    exit 1
fi
echo "✅ Configuration file: Found"

# Get the API URL from config
API_URL=$(php -r "require 'config.php'; echo getConfig('API_BASE_URL');")
echo "🔗 API Endpoint: $API_URL"

# Test API connectivity
echo ""
echo "🔍 Testing API connectivity..."
if curl -s --connect-timeout 5 "$API_URL/health" > /dev/null; then
    echo "✅ API server is reachable"
else
    echo "⚠️  Warning: API server may not be running"
    echo "   Make sure your FastAPI backend is started on $API_URL"
    echo ""
fi

# Set default port
PORT=${1:-8080}
HOST=${2:-localhost}

echo ""
echo "🌐 Starting PHP development server..."
echo "   URL: http://$HOST:$PORT"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the PHP server
php -S $HOST:$PORT
