// chat-interface.tsx
import React, { ReactNode } from 'react';

interface ScrollAreaProps {
  className?: string;
  children: ReactNode;
}

// interface Message {
//   agentName: string;
//   text: string;
//   avatar?: string;
//   replies?: number;
//   reactions?: string[];
// }

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
  }

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages }) => {
    const messagesEndRef = React.useRef<HTMLDivElement>(null);
    
    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    
    // Scroll to bottom when messages change
    React.useEffect(() => {
      scrollToBottom();
    }, [messages]);

    const parseMessage = (message: { text: string, type: 'message' | 'status' }): Message => {
        if (message.type === 'status') {
        return {
            agentName: '',
            text: message.text,
            type: 'status'
        };
        }
        const [agentName, text] = message.text.split(': ');
        return {
            agentName,
            text,
            type: 'message'
        };
    };

  
    return (
      <div id="chat-container" className="flex flex-col h-full w-full">
        <div id="chat-header">Chat</div>
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
      </div>
    );
};