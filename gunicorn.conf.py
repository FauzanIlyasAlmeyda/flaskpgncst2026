bind = "0.0.0.0:5000"
workers = 2
threads = 4
timeout = 600        # 10 menit — cukup untuk training Thorough
worker_class = "gthread"
accesslog = "-"
errorlog = "-"
loglevel = "info"
