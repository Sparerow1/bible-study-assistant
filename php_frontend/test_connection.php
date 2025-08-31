<?php
/**
 * BibleBot PHP Frontend Connection Test
 * 
 * This script tests the connection between the PHP frontend and FastAPI backend
 */

require_once 'config.php';

echo "🧪 BibleBot PHP Frontend Connection Test\n";
echo "========================================\n\n";

// Test 1: Configuration
echo "1. Testing Configuration...\n";
echo "   API Base URL: " . getConfig('API_BASE_URL') . "\n";
echo "   API Timeout: " . getConfig('API_TIMEOUT') . " seconds\n";
echo "   Session ID: " . getConfig('SESSION_ID') . "\n";
echo "   ✅ Configuration loaded successfully\n\n";

// Test 2: cURL Extension
echo "2. Testing cURL Extension...\n";
if (function_exists('curl_init')) {
    echo "   ✅ cURL extension is available\n";
} else {
    echo "   ❌ cURL extension is not available\n";
    exit(1);
}

// Test 3: API Health Check
echo "3. Testing API Health Check...\n";
$healthUrl = getConfig('API_BASE_URL') . '/health';
echo "   URL: $healthUrl\n";

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $healthUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, getConfig('API_TIMEOUT'));
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Accept: application/json']);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    echo "   ❌ cURL Error: $error\n";
    exit(1);
}

if ($httpCode !== 200) {
    echo "   ❌ HTTP Error: $httpCode\n";
    echo "   Response: $response\n";
    exit(1);
}

$healthData = json_decode($response, true);
if ($healthData && isset($healthData['status'])) {
    echo "   ✅ API Health Check: " . $healthData['status'] . "\n";
    if (isset($healthData['message'])) {
        echo "   Message: " . $healthData['message'] . "\n";
    }
    if (isset($healthData['vector_count'])) {
        echo "   Vector Count: " . $healthData['vector_count'] . "\n";
    }
} else {
    echo "   ⚠️  Unexpected response format\n";
    echo "   Response: $response\n";
}
echo "\n";

// Test 4: API Stats
echo "4. Testing API Stats...\n";
$statsUrl = getConfig('API_BASE_URL') . '/stats';
echo "   URL: $statsUrl\n";

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $statsUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, getConfig('API_TIMEOUT'));
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Accept: application/json']);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    echo "   ❌ cURL Error: $error\n";
} elseif ($httpCode !== 200) {
    echo "   ❌ HTTP Error: $httpCode\n";
    echo "   Response: $response\n";
} else {
    $statsData = json_decode($response, true);
    if ($statsData) {
        echo "   ✅ API Stats retrieved successfully\n";
        echo "   Total Vectors: " . ($statsData['total_vectors'] ?? 'N/A') . "\n";
        echo "   Dimension: " . ($statsData['dimension'] ?? 'N/A') . "\n";
        echo "   Index Name: " . ($statsData['index_name'] ?? 'N/A') . "\n";
    } else {
        echo "   ⚠️  Unexpected stats response format\n";
        echo "   Response: $response\n";
    }
}
echo "\n";

// Test 5: Chat Endpoint (Simple Test)
echo "5. Testing Chat Endpoint...\n";
$chatUrl = getConfig('API_BASE_URL') . '/chat';
echo "   URL: $chatUrl\n";

$testData = [
    'message' => 'Hello, this is a test message',
    'session_id' => getConfig('SESSION_ID')
];

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $chatUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, getConfig('API_TIMEOUT'));
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($testData));
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'Accept: application/json'
]);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    echo "   ❌ cURL Error: $error\n";
} elseif ($httpCode !== 200) {
    echo "   ❌ HTTP Error: $httpCode\n";
    echo "   Response: $response\n";
} else {
    $chatData = json_decode($response, true);
    if ($chatData && isset($chatData['answer'])) {
        echo "   ✅ Chat endpoint working\n";
        echo "   Response length: " . strlen($chatData['answer']) . " characters\n";
        if (isset($chatData['sources']) && is_array($chatData['sources'])) {
            echo "   Sources found: " . count($chatData['sources']) . "\n";
        }
    } else {
        echo "   ⚠️  Unexpected chat response format\n";
        echo "   Response: $response\n";
    }
}
echo "\n";

// Test 6: Clear Memory Endpoint
echo "6. Testing Clear Memory Endpoint...\n";
$clearUrl = getConfig('API_BASE_URL') . '/clear-memory';
echo "   URL: $clearUrl\n";

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $clearUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, getConfig('API_TIMEOUT'));
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Accept: application/json']);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($error) {
    echo "   ❌ cURL Error: $error\n";
} elseif ($httpCode !== 200) {
    echo "   ❌ HTTP Error: $httpCode\n";
    echo "   Response: $response\n";
} else {
    $clearData = json_decode($response, true);
    if ($clearData && isset($clearData['message'])) {
        echo "   ✅ Clear memory endpoint working\n";
        echo "   Message: " . $clearData['message'] . "\n";
    } else {
        echo "   ⚠️  Unexpected clear memory response format\n";
        echo "   Response: $response\n";
    }
}
echo "\n";

echo "🎉 Connection test completed!\n";
echo "If all tests passed, your PHP frontend should work correctly.\n";
echo "You can now start the PHP server with: ./start_server.sh\n";
?>
