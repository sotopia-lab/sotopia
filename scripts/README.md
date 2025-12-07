# Redis to Local Storage Migration Scripts

This directory contains scripts to migrate data from a Redis dump file to the local JSON storage format.

## Scripts

### `migrate_redis_to_local.sh` (Main Script)
The master script that orchestrates the entire migration process.

**Usage:**
```bash
./scripts/migrate_redis_to_local.sh
```

**What it does:**
1. Starts Redis with your dump file (`/Users/xuhuizhou/Downloads/dump_1.rdb`)
2. Exports all data to `~/.sotopia/data/` in local JSON format
3. Optionally stops Redis when done

### `start_redis_with_dump.sh`
Starts a Redis server with the dump file loaded.

**Usage:**
```bash
./scripts/start_redis_with_dump.sh
```

**Features:**
- Automatically detects and stops existing Redis instances on port 6379
- Uses `redis-stack-server` if available (required for dump files with Redis modules)
- Falls back to `redis-server` if redis-stack is not found
- Creates temporary directory for Redis data
- Waits for Redis to be ready before exiting

### `export_redis_to_local.py`
Python script to export all models from Redis to local JSON storage.

**Usage:**
```bash
# Basic usage (connects to redis://localhost:6379)
uv run python scripts/export_redis_to_local.py

# With custom Redis URL
uv run python scripts/export_redis_to_local.py --redis-url redis://localhost:6380

# With custom output directory
uv run python scripts/export_redis_to_local.py --output-dir /path/to/output

# Quiet mode (no progress bars)
uv run python scripts/export_redis_to_local.py --quiet
```

**Exported Models:**
- AgentProfile
- EnvironmentProfile
- RelationshipProfile
- EnvAgentComboStorage
- EnvironmentList
- Annotator
- EpisodeLog
- AnnotationForEpisode
- NonStreamingSimulationStatus

### `stop_redis.sh`
Stops the Redis server started by `start_redis_with_dump.sh`.

**Usage:**
```bash
./scripts/stop_redis.sh
```

## Output Format

Data is exported to `~/.sotopia/data/` with the following structure:

```
~/.sotopia/data/
├── AgentProfile/
│   ├── {uuid1}.json
│   ├── {uuid2}.json
│   └── ...
├── EnvironmentProfile/
│   ├── {uuid1}.json
│   └── ...
├── EnvAgentComboStorage/
│   ├── {uuid1}.json
│   └── ...
├── Annotator/
│   ├── {uuid1}.json
│   └── ...
└── ...
```

Each JSON file contains a single model instance with 2-space indentation:

```json
{
  "pk": "01H7VJPFPQ67TTMWZ9246SQ2A4",
  "env_id": "01H7VFHPJKR16MD1KC71V4ZRCF",
  "agent_ids": [
    "01H5TNE5PMBJ9VHH51YC0BB64C",
    "01H5TNE5P6KZKR2AEY6SZB83H0"
  ]
}
```

## Using the Exported Data

After running the migration, you can use the local storage backend by setting the environment variable:

```bash
export SOTOPIA_STORAGE_BACKEND=local
```

Then all Sotopia database operations will use the local JSON files instead of Redis:

```python
from sotopia.database import AgentProfile, Annotator

# Automatically uses local storage when SOTOPIA_STORAGE_BACKEND=local
annotators = Annotator.all()
for annotator in annotators:
    print(f"{annotator.name}: {annotator.email}")
```

## Requirements

- Redis 8.0+ or redis-stack-server (for loading dump files with Redis modules)
- Python 3.10+
- uv package manager
- Sotopia dependencies installed (`uv sync --all-extras`)

## Troubleshooting

### "Can't handle RDB format version X"
Your Redis version is too old. Upgrade to Redis 8.0+:
```bash
brew upgrade redis
```

### "The RDB file contains AUX module data I can't load"
The dump file was created with redis-stack-server (includes RediSearch, RedisJSON, etc.). Install redis-stack:
```bash
brew install redis-stack
```

### "Redis URL must specify one of the following schemes"
Make sure the `REDIS_OM_URL` environment variable is set correctly:
```bash
export REDIS_OM_URL="redis://localhost:6379"
```

## Migration Results

After successful migration, you should see output like:

```
Export Summary
============================================================
  ✓ Annotator: 3 records exported (0 errors)
  ✓ EnvAgentComboStorage: 450 records exported (0 errors)
  ✓ AnnotationForEpisode: 439 records exported (0 errors)
============================================================
Total: 892 records exported (0 errors)
```
