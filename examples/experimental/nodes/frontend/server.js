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
  // url: "redis://redis:6379"
});

redisClient.on('error', (err) => {
  console.error('Redis error:', err);
});

(async () => {
  await redisClient.connect();

  // Define allowed channels
  const allowedChannels = ['Scene:Jack', 'Scene:Jane', 'Human:Jack', 'Jack:Human', 'Agent:Runtime', 'Runtime:Agent'];

  // Subscribe only to allowed channels
  const subscriber = redisClient.duplicate();
  await subscriber.connect();
  await subscriber.subscribe(allowedChannels, (message, channel) => {
    console.log(`Received message from ${channel}: ${message}`);
    io.emit('new_message', { channel, message });
  });
})();

io.on('connection', (socket) => {
  console.log('A user connected');

  socket.on('chat_message', async (message) => {
    console.log('Received chat message:', message);
    try {
      // Send chat message to Jane:Jack channel
      // Send command to Runtime through Redis
      const agentAction = {
        data: {
          agent_name: "user",
          action_type: "speak",
          argument: message,
          path: "",
          data_type: "agent_action"
        }
      };
      await redisClient.publish('Human:Jack', JSON.stringify(agentAction));
    } catch (err) {
      console.error('Error publishing chat message:', err);
    }
  });



  // Listen for save_file event
  socket.on('save_file', async ({ path, content }) => {
    console.log('Saving file:', path);
    try {
      // Prepare the message to send to the Agent:Runtime channel
      const saveMessage = {
        data: {
          agent_name: "user",
          action_type: "write",
          argument: content,
          path: path,
          data_type: "agent_action"
        }
      };
      // Publish the message to the Agent:Runtime channel
      await redisClient.publish('Agent:Runtime', JSON.stringify(saveMessage));
    } catch (err) {
      console.error('Error publishing save file message:', err);
    }
  });


  socket.on('terminal_command', async (command) => {
    console.log('Received terminal command:', command);
    try {
      // Send command to Runtime through Redis
      const messageEnvelope = {
        data: {
          agent_name: "user",
          action_type: "run",
          argument: command,
          path: "",
          data_type: "agent_action"
        }
      };
      await redisClient.publish('Agent:Runtime', JSON.stringify(messageEnvelope));
    } catch (err) {
      console.error('Error publishing command:', err);
      socket.emit('new_message', {
        channel: 'Runtime:Agent',
        message: JSON.stringify({
          data: {
            data_type: "text",
            text: `Error: ${err.message}`
          }
        })
      });
    }
  });

  socket.on('disconnect', () => {
    console.log('A user disconnected');
  });
});

// Serve the index.html file
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/frontend/public/index.html');
});

// Start the server
server.listen(8000, () => {
  console.log('Server is running on http://localhost:8000');
});
