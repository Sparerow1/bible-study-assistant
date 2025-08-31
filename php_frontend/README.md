# BibleBot PHP Frontend

A PHP-based frontend for the BibleBot FastAPI chatbot that replicates the functionality of the original HTML interface.

## Features

- **Modern Chat Interface**: Beautiful, responsive chat UI with gradient backgrounds and smooth animations
- **Real-time Communication**: AJAX-based communication with the FastAPI backend
- **Health Monitoring**: Real-time status indicator and health checks
- **Statistics Display**: Shows vector count and dimension information
- **Memory Management**: Clear conversation memory functionality
- **Source References**: Displays biblical references and sources for answers
- **Error Handling**: Comprehensive error handling and user feedback
- **Configuration Management**: Easy configuration through `config.php`

## Requirements

- PHP 7.4 or higher
- cURL extension enabled
- Web server (Apache, Nginx, or PHP built-in server)
- FastAPI backend running (see main project README)

## Installation

1. **Clone or download the PHP frontend files** to your web server directory
2. **Configure the API endpoint** in `config.php`:
   ```php
   define('API_BASE_URL', 'http://localhost:8000'); // Change to your FastAPI server URL
   ```
3. **Ensure the FastAPI backend is running** (see main project documentation)
4. **Access the frontend** through your web browser

## Configuration

Edit `config.php` to customize the frontend:

### API Configuration
```php
define('API_BASE_URL', 'http://localhost:8000'); // FastAPI server URL
define('API_TIMEOUT', 30); // Request timeout in seconds
```

### UI Configuration
```php
define('CHAT_TITLE', 'BibleBot (PHP)');
define('CHAT_SUBTITLE', 'Your AI-powered Biblical Q&A Assistant');
```

### Security Settings
```php
define('MAX_MESSAGE_LENGTH', 1000); // Maximum message length
define('RATE_LIMIT_ENABLED', false); // Enable rate limiting
```

### Feature Flags
```php
define('ENABLE_STATS', true);
define('ENABLE_HEALTH_CHECK', true);
define('ENABLE_MEMORY_CLEAR', true);
define('ENABLE_SOURCES', true);
```

## Usage

### Starting the PHP Server

#### Option 1: PHP Built-in Server (Development)
```bash
cd php_frontend
php -S localhost:8080
```
Then visit `http://localhost:8080` in your browser.

#### Option 2: Apache/Nginx (Production)
1. Copy the `php_frontend` directory to your web server's document root
2. Configure your web server to serve PHP files
3. Access via your domain or IP address

### Using the Chat Interface

1. **Health Check**: The green dot in the top-right indicates API connectivity
2. **Statistics**: The top-left shows vector count and dimensions
3. **Chat**: Type your biblical questions in the input field
4. **Memory Clear**: Click "Clear Memory" to reset conversation context
5. **Sources**: Biblical references appear below bot responses

## File Structure

```
php_frontend/
├── index.php          # Main chat interface
├── config.php         # Configuration settings
└── README.md          # This file
```

## API Endpoints Used

The PHP frontend communicates with these FastAPI endpoints:

- `GET /health` - Health check and status
- `GET /stats` - Vector database statistics
- `POST /chat` - Send chat messages
- `POST /clear-memory` - Clear conversation memory

## Troubleshooting

### Common Issues

1. **"CURL Error"**: Ensure cURL extension is enabled in PHP
2. **"HTTP Error: 503"**: FastAPI backend is not running
3. **"HTTP Error: 404"**: Incorrect API_BASE_URL in config.php
4. **Empty responses**: Check FastAPI server logs for errors

### Debug Mode

Enable debug mode in `config.php`:
```php
define('DEBUG_MODE', true);
```

### Logging

Enable logging to track API requests:
```php
define('LOG_ENABLED', true);
define('LOG_FILE', 'biblebot_php.log');
```

## Security Considerations

- **Production Deployment**: Set `DEBUG_MODE` to `false`
- **CORS**: Configure `ALLOW_CORS` based on your deployment needs
- **Rate Limiting**: Enable rate limiting for production use
- **Input Validation**: Messages are validated for length and content
- **Error Handling**: Sensitive information is not exposed in error messages

## Customization

### Styling
Modify the CSS in `index.php` to customize the appearance:
- Colors and gradients
- Layout and spacing
- Animations and transitions
- Responsive design

### Functionality
Add new features by:
1. Adding new PHP functions in the backend section
2. Creating corresponding JavaScript functions
3. Updating the UI as needed

## Performance Tips

1. **Caching**: Consider implementing response caching for repeated questions
2. **Connection Pooling**: For high traffic, consider connection pooling
3. **CDN**: Serve static assets through a CDN
4. **Compression**: Enable gzip compression on your web server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This PHP frontend is part of the BibleBot project. See the main project license for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the FastAPI backend logs
3. Check the PHP error logs
4. Create an issue in the main project repository
