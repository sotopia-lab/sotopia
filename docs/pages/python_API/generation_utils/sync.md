# async_to_sync

Converts an asynchronous function into a synchronous function that can be called without an event loop. This is useful when you need to call async functions in a synchronous context.

## Parameters
- `async_func` (Callable[P, Awaitable[T]]): The asynchronous function to be converted.

## Returns
- `Callable[P, T]`: A synchronous version of the provided asynchronous function.

## Example Usage
```python
# Assuming agenerate is an asynchronous function that you need to call synchronously

# Synchronous wrapper for the async function
generate = async_to_sync(agenerate)

# Now, you can call `generate` synchronously
result = generate(arg1, arg2, ...)
```

# generate

A synchronous version of the `agenerate` asynchronous function.

## Example Usage
```python
result = generate(arg1, arg2, ...)
```

# generate_action

A synchronous version of the `agenerate_action` asynchronous function.

## Example Usage
```python
result = generate_action(arg1, arg2, ...)
```
