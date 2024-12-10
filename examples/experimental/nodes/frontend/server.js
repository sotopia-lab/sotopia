import { createServer } from 'http';
import { Server } from 'socket.io';
import { createClient } from 'redis';
import next from 'next';

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

// Redis client configuration
const redisClient = createClient({
  url: 'redis://localhost:6379/0'
});

// Allowed channels for Redis pub/sub 
const allowedChannels = ['Scene:Jack', 'Scene:Jane', 'Human:Jack', 'Jack:Human', 'Agent:Runtime', 'Runtime:Agent'];

app.prepare().then(async () => {
  // Connect Redis client
  redisClient.on('error', (err) => {
    console.error('Redis error:', err);
  });
  await redisClient.connect();

  // Create HTTP server
  const server = createServer((req, res) => {
    handle(req, res);
  });
  // Initialize Socket.IO server
  const io = new Server(server);

  // Redis subscriber setup
  const subscriber = redisClient.duplicate();
  await subscriber.connect();

  await subscriber.subscribe(allowedChannels, (message, channel) => {
    console.log(`Received message from ${channel}: ${message}`);
    io.emit('new_message', { channel, message });
  });

  // Socket.IO connection handling
  io.on('connection', (socket) => {
    console.log('A user connected');

    socket.on('chat_message', async (message) => {
      console.log('Received chat message:', message);
      try {
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

    socket.on('save_file', async ({ path, content }) => {
      console.log('Saving file:', path);
      try {
        const saveMessage = {
          data: {
            agent_name: "user",
            action_type: "write",
            argument: content,
            path: path,
            data_type: "agent_action"
          }
        };
        await redisClient.publish('Agent:Runtime', JSON.stringify(saveMessage));
      } catch (err) {
        console.error('Error publishing save file message:', err);
      }
    });

    socket.on('terminal_command', async (command) => {
      console.log('Received terminal command:', command);
      try {
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

  // Start the server
  const port = process.env.PORT || 3000;
  server.listen(port, (err) => {
    if (err) throw err;
    console.log(`> Ready on http://localhost:3000`);
  });
});

export default app;