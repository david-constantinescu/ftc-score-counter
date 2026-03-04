"""
Gunicorn configuration for Flask-SocketIO deployment.
Optimized for Render's free tier (0.1 CPU, 512MB RAM).
"""

import multiprocessing

# Worker configuration
# Calculate workers based on CPU cores (Render free tier: 0.1 CPU)
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class - REQUIRED for Flask-SocketIO
worker_class = 'gevent'

# Maximum concurrent connections per worker
worker_connections = 1000

# Timeout settings
timeout = 120  # Request timeout in seconds
keepalive = 5   # Keep connections alive for 5 seconds

# Graceful timeout
graceful_timeout = 30

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'
capture_output = True
enable_stdio_inheritance = True

# Process naming
proc_name = 'ftc-management'

# Worker reload settings
max_requests = 1000
max_requests_jitter = 50

# Bind settings (will be overridden by $PORT on Render)
bind = '0.0.0.0:24543'

# Worker threads (for gevent)
threads = 1
