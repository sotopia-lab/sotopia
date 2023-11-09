from redis_om import JsonModel, get_redis_connection

class Person(JsonModel):
    name: str
    age: int

# Create an instance of your model
person = Person(name="John", age=30)

# Save to Redis with a specific key
person.save()