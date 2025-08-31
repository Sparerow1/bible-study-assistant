<?php
// BibleBot PHP Frontend Configuration

// FastAPI Server Configuration
define('API_BASE_URL', 'http://localhost:8000'); // Change this to your FastAPI server URL
define('API_TIMEOUT', 30); // Request timeout in seconds

// Session Configuration
define('SESSION_ID', 'php_session');

// Error Reporting (set to false in production)
define('DEBUG_MODE', true);

// CORS Headers (if needed)
define('ALLOW_CORS', true);

// Security Settings
define('MAX_MESSAGE_LENGTH', 1000); // Maximum message length
define('RATE_LIMIT_ENABLED', false); // Enable rate limiting
define('RATE_LIMIT_REQUESTS', 10); // Requests per minute
define('RATE_LIMIT_WINDOW', 60); // Time window in seconds

// UI Configuration
define('CHAT_TITLE', 'BibleBot (PHP)');
define('CHAT_SUBTITLE', 'Your AI-powered Biblical Q&A Assistant');

// Feature Flags
define('ENABLE_STATS', true);
define('ENABLE_HEALTH_CHECK', true);
define('ENABLE_MEMORY_CLEAR', true);
define('ENABLE_SOURCES', true);

// Logging Configuration
define('LOG_ENABLED', false);
define('LOG_FILE', 'biblebot_php.log');

// Helper function to get configuration
function getConfig($key, $default = null) {
    if (defined($key)) {
        return constant($key);
    }
    return $default;
}

// Helper function to log messages
function logMessage($message, $level = 'INFO') {
    if (!getConfig('LOG_ENABLED')) {
        return;
    }
    
    $logFile = getConfig('LOG_FILE');
    $timestamp = date('Y-m-d H:i:s');
    $logEntry = "[$timestamp] [$level] $message" . PHP_EOL;
    
    file_put_contents($logFile, $logEntry, FILE_APPEND | LOCK_EX);
}

// Helper function to validate message
function validateMessage($message) {
    if (empty($message)) {
        return ['valid' => false, 'error' => 'Message cannot be empty'];
    }
    
    if (strlen($message) > getConfig('MAX_MESSAGE_LENGTH')) {
        return ['valid' => false, 'error' => 'Message too long'];
    }
    
    return ['valid' => true];
}

// Helper function to set CORS headers
function setCorsHeaders() {
    if (getConfig('ALLOW_CORS')) {
        header('Access-Control-Allow-Origin: *');
        header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
        header('Access-Control-Allow-Headers: Content-Type');
    }
}

// Helper function to handle preflight requests
function handlePreflight() {
    if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
        setCorsHeaders();
        http_response_code(200);
        exit;
    }
}
?>
