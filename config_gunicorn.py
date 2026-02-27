
# Number of worker processes
workers = 2  

# Worker class (Uvicorn for FastAPI)
worker_class = "uvicorn.workers.UvicornWorker"

# Host and port, localisation of uvicorn
bind = "127.0.0.1:8000"

# Logging
accesslog = "-"  # stdout
errorlog = "-"
