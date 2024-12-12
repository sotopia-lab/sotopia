# Use an official Node.js runtime as the base image
FROM node:20-alpine

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Redis client
RUN apk add --no-cache redis

# Copy package.json and package-lock.json
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the rest of the application code
COPY server.js ./

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["node", "server.js"]