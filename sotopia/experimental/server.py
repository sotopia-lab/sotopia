from typing import AsyncGenerator, Any
import asyncio
import redis
import json
from sotopia.experimental.envs import generate_executable
import uuid
import logging
from rich import print
import os  # Add import for os module
from rich.logging import RichHandler

# Configure logger
logger = logging.getLogger("sotopia.experimental.server")
logger.setLevel(logging.INFO)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


async def arun_one_episode(
    episode_config: dict[str, Any],
    connection_id: str = str(uuid.uuid4()),  # the connection id for the websocket
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run one episode of simulation and yield messages as they are generated.

    Returns:
        AsyncGenerator yielding simulation messages
    """
    episode_config["connection_id"] = connection_id
    episode_config["agents"] += [
        {
            "name": "redis_agent",
        }
    ]
    # Generate the executable config and save it to a temporary file
    executable_config_content = generate_executable(episode_config)

    # Create a unique temporary filename
    temp_filename = f"temp_config_{connection_id}.toml"

    # Write the config to the temporary file
    with open(temp_filename, "w") as f:
        f.write(executable_config_content)

    executable_config = temp_filename

    # Connect to Redis using the async client
    redis_url = episode_config.get("redis_url", "redis://localhost:6379/0")
    redis_client = redis.asyncio.from_url(redis_url)
    pubsub = redis_client.pubsub()
    channel = f"{connection_id}"
    logger.info(f"Subscribing to channel: {channel}")
    await pubsub.subscribe(channel)

    # Run the dataflow
    run_cmd = f"aact run-dataflow {executable_config}"
    # Start the process and capture stdout and stderr for debugging
    proc = await asyncio.create_subprocess_shell(
        run_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Create tasks to read and log stdout and stderr without blocking
    async def log_stdout() -> None:
        while True:
            if proc.stdout is not None:
                line = await proc.stdout.readline()
            else:
                line = b""
            if not line:
                break
            print(f"{line.decode().strip()}")

    async def log_stderr() -> None:
        while True:
            if proc.stderr is not None:
                err = await proc.stderr.readline()
            else:
                err = b""
            if not err:
                break
            print(f"{err.decode().strip()}")

    # Start the logging tasks
    stdout_task = asyncio.create_task(log_stdout())
    stderr_task = asyncio.create_task(log_stderr())

    # Connect to Redis using the async client
    redis_client = redis.asyncio.Redis(host="localhost", port=6379, db=0)
    pubsub = redis_client.pubsub()
    channel = f"{connection_id}" if connection_id else "sotopia:simulation"
    print(f"Subscribing to channel: {channel}")
    await pubsub.subscribe(channel)

    # Create a task to monitor the process completion
    process_done = asyncio.Event()

    async def monitor_process() -> None:
        await proc.wait()
        process_done.set()

    monitor_task = asyncio.create_task(monitor_process())

    try:
        # Process Redis messages until the dataflow process is done
        while not process_done.is_set():
            # Use wait_for with a timeout to periodically check if process is done
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True), timeout=0.1
                )
                if message is None:
                    # No message received within timeout, check if process is done
                    continue

                # Process the message data
                try:
                    line_str = message["data"].decode()
                    message_data = json.loads(line_str)
                    potential_episode_log = json.loads(
                        message_data.get("last_turn", "{}")
                    )
                    if "messages" in potential_episode_log:
                        yield potential_episode_log
                    else:
                        # Handle other message types that don't have the expected format
                        logger.debug(
                            f"Received message with unexpected format: {message_data}"
                        )

                except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                    logger.error(f"Error processing message: {e}")
                    continue

            except asyncio.TimeoutError:
                # Timeout occurred, loop will continue and check if process is done
                continue

    except asyncio.CancelledError:
        # Handle cancellation
        raise
    finally:
        # Clean up Redis connection
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_client.close()

        # Cancel monitoring tasks
        monitor_task.cancel()
        stdout_task.cancel()
        stderr_task.cancel()

        try:
            await asyncio.gather(
                monitor_task, stdout_task, stderr_task, return_exceptions=True
            )
        except asyncio.CancelledError:
            pass

        # Terminate the process if it's still running
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()

        # Remove the temporary configuration file
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"Removed temporary configuration file: {temp_filename}")
        except Exception as e:
            logger.error(f"Failed to remove temporary file {temp_filename}: {e}")

        # Check for errors
        if proc.returncode and proc.returncode != 0:
            stderr_content = await proc.stderr.read()
            raise RuntimeError(f"Dataflow execution failed: {stderr_content.decode()}")
