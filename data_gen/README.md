# Data Generation

For the first step, we generate envProfile (including scenario / social goal / relationship restriction) based on inspiring prompt.

For the 2.1 step, we put the original agentProfile and relationshipProfile into our new redis database

For the 2.2 step, we combine them together to be combos based on conditiona sampling (the restriction is the relationship)

All the EnvProfile (new generated), AgentProfile (sotopia original), RelationshipProfile (sotopia original), and envagentcombo are on the redis database that is new created.

For the third step, we need to use another version of redis and convert it into json file and save the whole data in the database on the local machine.

For the final step, we convert the whole thing into Ruiyi's format.

# Local Redis Setting
Since the redis-server cannot directly input json data, it requires loading a RedisJson model into the redis-server to enable this function. Therefore, we need to load a docker based on RedisJson:

docker run -p 6379:6379 --name redis-stack redis/redis-stack:latest

Link: <https://github.com/RedisJSON/RedisJSON>
