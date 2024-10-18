import redis
import time
import threading

def publish_message():
    # Initialize Redis client
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Publish a message to the channel
    time.sleep(1)  # Wait for a moment to ensure the subscriber is ready
    redis_client.publish('test_channel', 'Hello, Redis!')

def subscribe_and_receive():
    # Initialize Redis client and subscribe to a channel
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('test_channel')
    
    print("Subscribed to channel 'test_channel'. Waiting for messages...")
    
    # Continuously check for messages
    while True:
        message = pubsub.get_message(ignore_subscribe_messages=True)
        if message:
            print(f"Received message: {message['data'].decode('utf-8')}")
            break  # Exit after receiving the message
        time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

if __name__ == "__main__":
    # Start the subscriber in a separate thread
    subscriber_thread = threading.Thread(target=subscribe_and_receive)
    subscriber_thread.start()
    
    # Publish a message
    publish_message()
    
    # Wait for the subscriber thread to finish
    subscriber_thread.join()