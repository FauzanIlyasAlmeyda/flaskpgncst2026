import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = 2
threads = 4
timeout = 600
worker_class = "gthread"
accesslog = "-"
errorlog = "-"
loglevel = "info"
