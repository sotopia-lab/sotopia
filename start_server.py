
import os
import sys

# Set up path
sys.path.insert(0, os.getcwd())

# Initialize Redis OM explicitly
from redis_om import get_redis_connection
redis = get_redis_connection(host='localhost', port=6379)
print(f"Redis connection successful: {redis.ping()}")

# Now import and run the server
from sotopia.api.fastapi_server import app
import uvicorn
uvicorn.run(app, host='localhost', port=8800)
