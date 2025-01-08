import modal
import subprocess
import time
import os

import redis
from sotopia.api.fastapi_server import SotopiaFastAPI

# Create persistent volume for Redis data
redis_volume = modal.Volume.from_name("sotopia-api", create_if_missing=True)


def initialize_redis_data() -> None:
    """Download Redis data if it doesn't exist"""
    if not os.path.exists("/vol/redis/dump.rdb"):
        os.makedirs("/vol/redis", exist_ok=True)
        print("Downloading initial Redis data...")
        subprocess.run(
            "curl -L https://cmu.box.com/shared/static/e3vd31r7916jb70j9cgtcq9etryrxml0.rdb --output /vol/redis/dump.rdb",
            shell=True,
            check=True,
        )
        print("Redis data downloaded")


# Create image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "gpg",
        "lsb-release",
        "wget",
        "procps",  # for ps command
        "redis-tools",  # for redis-cli
    )
    .run_commands(
        # Update and install basic dependencies
        "apt-get update",
        "apt-get install -y curl gpg lsb-release",
        # Add Redis Stack repository
        "curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg",
        "chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg",
        'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list',
        "apt-get update",
        "apt-get install -y redis-stack-server",
    )
    .pip_install(
        "sotopia>=0.1.2",
        "fastapi>0.100",  # TODO: remove this dependency after pypi release
        "uvicorn",  # TODO: remove this dependency after pypi release
    )
)
redis_volume = modal.Volume.from_name("sotopia-api", create_if_missing=True)

# Create stub for the application
app = modal.App("sotopia-fastapi", image=image, volumes={"/vol/redis": redis_volume})


@app.cls(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=5,
    secrets=[modal.Secret.from_name("openai-secret")],
)
class WebAPI:
    def __init__(self) -> None:
        self.web_app = SotopiaFastAPI()

    @modal.enter()
    def setup(self) -> None:
        # Start Redis server
        subprocess.Popen(
            ["redis-stack-server", "--dir", "/vol/redis", "--port", "6379"]
        )

        # Wait for Redis to be ready
        max_retries = 30
        for _ in range(max_retries):
            try:
                initialize_redis_data()
                # Attempt to create Redis client and ping the server
                temp_client = redis.Redis(host="localhost", port=6379, db=0)
                temp_client.ping()
                self.redis_client = temp_client
                print("Successfully connected to Redis")
                return
            except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError):
                print("Waiting for Redis to be ready...")
                time.sleep(1)

        raise Exception("Could not connect to Redis after multiple attempts")

    @modal.exit()
    def cleanup(self) -> None:
        if hasattr(self, "redis_client"):
            self.redis_client.close()

    @modal.asgi_app()
    def serve(self) -> SotopiaFastAPI:
        return self.web_app
