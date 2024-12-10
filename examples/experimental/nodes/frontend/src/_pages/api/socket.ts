import { NextApiRequest, NextApiResponse } from 'next';
import { Server } from 'socket.io';
import { createClient } from 'redis';
import { Server as HttpServer } from 'http';

// Extend NextApiResponse to include socket server
interface NextApiResponseWithSocket extends NextApiResponse {
  socket: NextApiResponse['socket'] & {
    server: HttpServer & {
      io?: Server;
    };
  };
}

const redisConfig = {
  url: 'redis://localhost:6379/0'
};

const allowedChannels = [
  'Scene:Jack', 'Scene:Jane', 
  'Human:Jack', 'Jack:Human', 
  'Agent:Runtime', 'Runtime:Agent'
];

export default async function handler(
  _req: NextApiRequest, 
  res: NextApiResponseWithSocket
) {
  // Handle WebSocket connection
  if (res.socket.server.io) {
    res.end();
    return;
  }

  console.log("Starting Socket.IO server on port:", 3000)

  const io = new Server({ path: "/api/socket", addTrailingSlash: false, cors: { origin: "*" } }).listen(3000)
  res.socket.server.io = io;

  console.log('server built')

  const redisClient = createClient(redisConfig);
  const redisSubscriber = redisClient.duplicate();

  try {
    await redisClient.connect();
    await redisSubscriber.connect();
    console.log('redis built')

    await redisSubscriber.subscribe(allowedChannels, (message, channel) => {
      io.emit('new_message', { channel, message });
    });

    io.on('connection', (socket) => {
      console.log('Client connected');

      socket.on('chat_message', async (message) => {
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
          console.error('Chat message publish error:', err);
        }
      });

      socket.on('save_file', async ({ path, content }) => {
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
          console.error('File save error:', err);
        }
      });

      socket.on('terminal_command', async (command) => {
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
          console.error('Terminal command error:', err);
        }
      });
    });
    
    res.status(200).json({ message: 'Success' })
    res.end();
  } catch (err) {
    console.error('Socket initialization error:', err);
    res.status(500).json({ error: 'Socket initialization failed' });
  }
}