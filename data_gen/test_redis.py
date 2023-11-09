import redis

r = redis.Redis(
  host='us1-normal-burro-37804.upstash.io',
  port=37804,
  password='a870a438f928424bb507d5895b3ab3fc'
)

r.set('foo', 'bar')
print(r.get('foo'))