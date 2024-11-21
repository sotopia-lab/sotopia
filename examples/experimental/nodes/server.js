// server.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { createClient } = require('redis');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Connect to Redis
const redisClient = createClient({
  url: 'redis://localhost:6379/0'
});

redisClient.on('error', (err) => {
  console.error('Redis error:', err);
});

(async () => {
  await redisClient.connect();

  // Define allowed channels
  const allowedChannels = ['Scene:Jack', 'Scene:Jane', 'Jane:Jack', 'Jack:Jane', 'Agent:Runtime', 'Runtime:Agent'];

  // Subscribe only to allowed channels
  const subscriber = redisClient.duplicate();
  await subscriber.connect();
  await subscriber.subscribe(allowedChannels, (message, channel) => {
    console.log(`Received message from ${channel}: ${message}`);
    io.emit('new_message', { channel, message });
  });
})();

// Serve the index.html file
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/frontend/public/index.html');
});

// Start the server
server.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});