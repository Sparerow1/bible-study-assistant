<?php
// Include configuration
require_once 'config.php';

// Handle preflight requests
handlePreflight();

// Handle AJAX requests
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action'])) {
    header('Content-Type: application/json');
    
    switch ($_POST['action']) {
        case 'chat':
            handleChatRequest();
            break;
        case 'health':
            handleHealthCheck();
            break;
        case 'stats':
            handleStatsRequest();
            break;
        case 'clear_memory':
            handleClearMemory();
            break;
        default:
            echo json_encode(['error' => 'Invalid action']);
    }
    exit;
}

function handleChatRequest() {
    $message = $_POST['message'] ?? '';
    
    // Validate message
    $validation = validateMessage($message);
    if (!$validation['valid']) {
        echo json_encode(['error' => $validation['error']]);
        return;
    }
    
    $data = [
        'message' => $message,
        'session_id' => $_POST['session_id'] ?? getConfig('SESSION_ID')
    ];
    
    $response = makeApiRequest(getConfig('API_BASE_URL') . '/chat', $data);
    echo $response;
}

function handleHealthCheck() {
    $response = makeApiRequest(getConfig('API_BASE_URL') . '/health', null, 'GET');
    echo $response;
}

function handleStatsRequest() {
    $response = makeApiRequest(getConfig('API_BASE_URL') . '/stats', null, 'GET');
    echo $response;
}

function handleClearMemory() {
    $response = makeApiRequest(getConfig('API_BASE_URL') . '/clear-memory', null, 'POST');
    echo $response;
}

function makeApiRequest($url, $data = null, $method = 'POST') {
    $ch = curl_init();
    
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, getConfig('API_TIMEOUT'));
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Accept: application/json'
    ]);
    
    if ($method === 'POST') {
        curl_setopt($ch, CURLOPT_POST, true);
        if ($data) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        }
    }
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    curl_close($ch);
    
    // Log the request
    logMessage("API Request: $method $url - HTTP Code: $httpCode");
    
    if ($error) {
        logMessage("CURL Error: $error", 'ERROR');
        return json_encode(['error' => 'CURL Error: ' . $error]);
    }
    
    if ($httpCode !== 200) {
        logMessage("HTTP Error: $httpCode", 'ERROR');
        return json_encode(['error' => 'HTTP Error: ' . $httpCode]);
    }
    
    return $response;
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo getConfig('CHAT_TITLE'); ?> - 圣经学习助手</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 800px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .header p {
            opacity: 0.9;
            font-size: 14px;
        }

        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 20px;
            position: relative;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }

        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 5px;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin: 0 10px;
        }

        .user .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .bot .message-avatar {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }

        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }

        .chat-input input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input input:focus {
            border-color: #667eea;
        }

        .chat-input button {
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }

        .chat-input button:hover {
            transform: translateY(-2px);
        }

        .chat-input button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #666;
            font-style: italic;
        }

        .loading-dots {
            display: flex;
            gap: 5px;
        }

        .loading-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: loading 1.4s infinite ease-in-out;
        }

        .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes loading {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .sources {
            margin-top: 10px;
            padding: 10px;
            background: #f0f8ff;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .sources h4 {
            color: #667eea;
            margin-bottom: 5px;
            font-size: 14px;
        }

        .source-item {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            padding: 5px;
            background: white;
            border-radius: 5px;
        }

        .error {
            color: #d32f2f;
            background: #ffebee;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
        }

        .stats {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }

        .clear-button {
            position: absolute;
            top: 20px;
            right: 50px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }

        .clear-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <button class="clear-button" onclick="clearMemory()">清空记忆</button>
            <h1>📖 <?php echo getConfig('CHAT_TITLE'); ?></h1>
            <p><?php echo getConfig('CHAT_SUBTITLE'); ?></p>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    你好！我是你的圣经学习助手。我在这里帮助你探索和理解圣经。问我任何关于圣经、神学或灵修的问题！
                </div>
            </div>
        </div>

        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="问我关于圣经、神学或灵修的问题..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()" id="sendButton">发送</button>
        </div>
    </div>

    <script>
        let isProcessing = false;

        // Initialize the chat
        document.addEventListener('DOMContentLoaded', function() {
            checkHealth();
            loadStats();
        });

        async function checkHealth() {
            try {
                const formData = new FormData();
                formData.append('action', 'health');
                
                const response = await fetch('', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                const statusIndicator = document.getElementById('statusIndicator');
                if (data.status === 'healthy') {
                    statusIndicator.style.background = '#4CAF50';
                } else {
                    statusIndicator.style.background = '#f44336';
                }
            } catch (error) {
                console.error('Health check failed:', error);
                document.getElementById('statusIndicator').style.background = '#f44336';
            }
        }

        async function loadStats() {
            try {
                const formData = new FormData();
                formData.append('action', 'stats');
                
                const response = await fetch('', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('stats').innerHTML = 'Stats unavailable';
                } else {
                    document.getElementById('stats').innerHTML = 
                        `📊 ${data.total_vectors} vectors<br>📏 ${data.dimension}d`;
                }
            } catch (error) {
                console.error('Failed to load stats:', error);
                document.getElementById('stats').innerHTML = 'Stats unavailable';
            }
        }

        async function clearMemory() {
            try {
                const formData = new FormData();
                formData.append('action', 'clear_memory');
                
                const response = await fetch('', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Failed to clear memory: ' + data.error);
                } else {
                    alert('Memory cleared successfully!');
                }
            } catch (error) {
                console.error('Failed to clear memory:', error);
                alert('Failed to clear memory');
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !isProcessing) {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || isProcessing) return;

            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';

            // Show loading message
            const loadingId = addLoadingMessage();

            try {
                isProcessing = true;
                document.getElementById('sendButton').disabled = true;

                const formData = new FormData();
                formData.append('action', 'chat');
                formData.append('message', message);
                formData.append('session_id', 'php_session');

                const response = await fetch('', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // Remove loading message
                removeMessage(loadingId);
                
                if (data.error) {
                    addErrorMessage('Sorry, I encountered an error while processing your request. Please try again.');
                } else {
                    // Add bot response
                    addBotMessage(data.answer, data.sources);
                }

            } catch (error) {
                console.error('Error sending message:', error);
                removeMessage(loadingId);
                addErrorMessage('Sorry, I encountered an error while processing your request. Please try again.');
            } finally {
                isProcessing = false;
                document.getElementById('sendButton').disabled = false;
                input.focus();
            }
        }

        function addMessage(content, type) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = type === 'user' ? '👤' : '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = content;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageDiv.id;
        }

        function addBotMessage(content, sources) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\n/g, '<br>');
            
            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<h4>📚 Biblical References:</h4>';
                
                sources.forEach((source, index) => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    sourceItem.textContent = `${index + 1}. ${source.content}`;
                    sourcesDiv.appendChild(sourceItem);
                });
                
                messageContent.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addLoadingMessage() {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            messageDiv.id = 'loading-message';
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = `
                <div class="loading">
                    <span>Thinking</span>
                    <div class="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return 'loading-message';
        }

        function removeMessage(messageId) {
            const message = document.getElementById(messageId);
            if (message) {
                message.remove();
            }
        }

        function addErrorMessage(content) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = `<div class="error">${content}</div>`;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    </script>
</body>
</html>
