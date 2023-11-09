from rejson import Client, Path
import json

redis_host = 'localhost'
redis_port = 6379
redis_password = ''

rj = Client(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def get_redisjson_value(key):
    try:
        return rj.jsonget(key, Path.rootPath())
    except Exception as e:
        print(f"Could not retrieve JSON for key {key}: {e}")
        return None

cursor = '0'
all_json_data = {}
while cursor != 0:
    cursor, keys = rj.scan(cursor=cursor, match='*')
    for key in keys:
        key_type = rj.type(key)
        if key_type == 'ReJSON-RL':
            json_value = get_redisjson_value(key)
            if json_value is not None:
                all_json_data[key] = json_value
        else:
            print(f"Key {key} is not of type ReJSON-RL, it's type is {key_type}")

with open('redis_json_data.json', 'w') as f:
    json.dump(all_json_data, f, indent=4)