import './ChatInterface.css';
import React, { ReactNode, useState, useEffect } from 'react';
import { Socket } from 'socket.io-client'; // Import the Socket type
import { FaComments } from 'react-icons/fa'; // Import the chat icon


interface ScrollAreaProps {
  className?: string;
  children: ReactNode;
}

interface Message {
  agentName: string;
  text: string;
  type: 'message' | 'status';
}

const ScrollArea: React.FC<ScrollAreaProps> = ({ children, className }) => {
  return (
    <div className={`overflow-y-auto ${className}`} id="messages">
      {children}
    </div>
  );
};

interface ChatInterfaceProps {
  messages: Array<{
    text: string;
    type: 'message' | 'status';
  }>;
  socket: Socket;
  onSendMessage: (text: string) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  socket,
  onSendMessage
}) => {
  const [input, setInput] = useState('');
  const [showBlinkingLight, setShowBlinkingLight] = useState(false);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  console.log("ChatInterface received messages:", messages);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

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

  const handleSend = () => {
    if (input.trim()) {
      socket.emit('chat_message', input.trim());
      onSendMessage(input.trim());
      setInput('');
    }
  };

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
        {showBlinkingLight && <div className="blinking-light"></div>}
      </div>
      <ScrollArea className="flex-grow">
        {messages.map((message, index) => {
          const parsedMessage = parseMessage(message);
          return parsedMessage.type === 'status' ? (
            <div key={index} className="status-message">
              {parsedMessage.text}
            </div>
          ) : (
            <div key={index} className="message" data-sender={parsedMessage.agentName}>
              <div className="message-avatar">
                <div className="avatar-placeholder">
                  {parsedMessage.agentName[0].toUpperCase()}
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
        <div ref={messagesEndRef} />
      </ScrollArea>

      <div className="chat-input-container p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            className="flex-grow p-2 rounded-lg border"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type a message..."
          />
          <button
            onClick={handleSend}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Send
          </button>
        </div>
      </div>

    </div>
  );
};
