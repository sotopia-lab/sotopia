# Q&A
## Missing episodes

Large batch size may cause some episodes to be skipped. This is due to the fact that the server may not be able to handle the load. Try reducing the batch size. But you can also use the script in `examples/fix_missing_episodes.py` to fix the missing episodes.

## Where I can find the data?

For the full data:
```sh
mkdir ~/redis-data
curl -L https://cmu.box.com/shared/static/xiivc5z8rnmi1zr6vmk1ohxslylvynur --output ~/redis-data/dump.rdb
```

For the data with only agents and their relationships:
```sh
mkdir ~/redis-data
curl -L https://cmu.box.com/s/9s7ooi9chpavjgqfjrpwzywp409j6ntr --output ~/redis-data/dump.rdb
```

Then you can start your database with:
```sh
sudo docker run -d -e REDIS_ARGS="--requirepass QzmCUD3C3RdsR" --name redis-stack -p 6379:6379 -p 8001:8001 -v /home/ubuntu/redis-data/:/data/ redis/redis-stack:latest
```

Redis saves snapshots of the database every few minutes. You can find them in the corresponding folder (for example, `\data\dump.rdb`). Use the `sudo docker cp <container_id>:/data/dump.rdb /tmp` to obtain the snapshot from the container (in this case, the copied data is stored in `/tmp/snap-private-tmp/snap.docker/tmp`).
