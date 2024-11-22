/**
 * ChatInterface.tsx
 *
 * This component represents a chat interface within the application. It allows users to
 * send and receive messages in real-time. The chat interface communicates with the server
 * via WebSocket to send user messages and receive messages from other agents or users.
 *
 * Key Features:
 * - Displays a list of messages with sender information.
 * - Supports sending messages via an input field and a send button.
 * - Automatically scrolls to the bottom when new messages are received.
 * - Displays a blinking light indicator when new messages arrive.
 *
 * Props:
 * - messages: An array of message objects to be displayed in the chat.
 * - socket: The WebSocket connection used to send messages to the server.
 * - onSendMessage: A callback function to handle the sending of messages.
 *
 */

import './ChatInterface.css';
import React, { ReactNode, useState, useEffect } from 'react';
import { Socket } from 'socket.io-client'; // Import the Socket type
import { FaComments } from 'react-icons/fa'; // Import the chat icon

// Define the props for the scrollable area
interface ScrollAreaProps {
  className?: string;
  children: ReactNode;
}

// Define the structure of a message
interface Message {
  agentName: string;
  text: string;
  type: 'message' | 'status';
}

// Scrollable area component for displaying messages
const ScrollArea: React.FC<ScrollAreaProps> = ({ children, className }) => {
  return (
    <div className={`overflow-y-auto ${className}`} id="messages">
      {children}
    </div>
  );
};

// Define the props for the ChatInterface component
interface ChatInterfaceProps {
  messages: Array<{
    text: string;
    type: 'message' | 'status';
  }>;
  socket: Socket;
  onSendMessage: (text: string) => void;
}

// Main ChatInterface component definition
export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  socket,
  onSendMessage
}) => {
  const [input, setInput] = useState(''); // State for the input field
  const [showBlinkingLight, setShowBlinkingLight] = useState(false); // State for blinking light indicator
  const messagesEndRef = React.useRef<HTMLDivElement>(null); // Ref for scrolling to the bottom

  console.log("ChatInterface received messages:", messages);

  // Scroll to the bottom of the chat when new messages are received
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Effect to handle scrolling and blinking light when messages change
  useEffect(() => {
    scrollToBottom();
    if (messages.length > 0) {
      setShowBlinkingLight(true);
      const timer = setTimeout(() => {
        setShowBlinkingLight(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [messages]);

  // Handle sending a message
  const handleSend = () => {
    if (input.trim()) {
      socket.emit('chat_message', input.trim()); // Send message via socket
      onSendMessage(input.trim()); // Call the provided callback
      setInput(''); // Clear the input field
    }
  };

  // Parse a message object into a structured format
  const parseMessage = (message: { text: string, type: 'message' | 'status' }): Message => {
    if (message.type === 'status') {
      return {
        agentName: '',
        text: message.text,
        type: 'status'
      };
    }
    const colonIndex = message.text.indexOf(':');
    if (colonIndex === -1) {
      return {
        agentName: 'System',
        text: message.text,
        type: 'message'
      };
    }
    const agentName = message.text.slice(0, colonIndex);
    const text = message.text.slice(colonIndex + 1).trim();
    return {
      agentName,
      text,
      type: 'message'
    };
  };

  return (
    <div id="chat-container" className="flex flex-col h-full w-full">
      <div id="chat-header" className="flex items-center justify-between">
        <div className="flex items-center">
          <FaComments className="chat-icon" size={12} />
          Chat
        </div>
        {showBlinkingLight && <div className="blinking-light"></div>} {/* Blinking light for new messages */}
      </div>
      <ScrollArea className="flex-grow">
        {messages.map((message, index) => {
          const parsedMessage = parseMessage(message); // Parse the message
          return parsedMessage.type === 'status' ? (
            <div key={index} className="status-message">
              {parsedMessage.text}
            </div>
          ) : (
            <div key={index} className="message" data-sender={parsedMessage.agentName}>
              <div className="message-avatar">
                <div className="avatar-placeholder">
                  {parsedMessage.agentName[0].toUpperCase()} {/* Display first letter of agent's name */}
                </div>
              </div>
              <div className="message-content-wrapper">
                <span className="message-sender">
                  {parsedMessage.agentName}
                </span>
                <div className="message-bubble">
                  {parsedMessage.text}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} /> {/* Reference for scrolling */}
      </ScrollArea>

      <div className="chat-input-container p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            className="flex-grow p-2 rounded-lg border"
            value={input}
            onChange={(e) => setInput(e.target.value)} // Update input state
            onKeyPress={(e) => e.key === 'Enter' && handleSend()} // Send message on Enter key
            placeholder="Type a message..."
          />
          <button
            onClick={handleSend} // Send message on button click
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Send
          </button>
        </div>
      </div>

    </div>
  );
};
