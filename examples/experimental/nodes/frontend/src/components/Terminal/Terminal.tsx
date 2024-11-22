import React, { useState, useRef, useEffect } from 'react';
import { Socket } from 'socket.io-client'; // Import the Socket type
import './Terminal.css'; // Import the CSS file
import { FaTerminal } from 'react-icons/fa'; // Import the terminal icon

interface TerminalProps {
  externalMessages: string[];
  socket: Socket;
}

// Function to strip ANSI color codes
const stripAnsiCodes = (text: string): string => {
  return text.replace(/\[\d+(?:;\d+)*m|\[\d+m|\[0m|\[1m/g, '');
};

// Function to convert ANSI codes to CSS classes
const processTerminalLine = (line: string): JSX.Element => {
  if (line.startsWith('Requirement already satisfied:')) {
    return <div style={{ color: '#00ff00' }}>{stripAnsiCodes(line)}</div>;
  }
  if (line.includes('notice')) {
    return <div style={{ color: '#808080' }}>{stripAnsiCodes(line)}</div>;
  }
  return <div>{stripAnsiCodes(line)}</div>;
};

export const Terminal: React.FC<TerminalProps> = ({ externalMessages, socket }) => {
  const [history, setHistory] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [history]);

  useEffect(() => {
    if (externalMessages.length > 0) {
      // Append the latest message to history
      setHistory((prevHistory) => [
        ...prevHistory,
        externalMessages[externalMessages.length - 1],
      ]);
    }
  }, [externalMessages]);

  const handleCommand = (command: string) => {
    setInput('');
    socket.emit('terminal_command', command); // Emit the command to the server
    // Optionally, you can add the command to history if desired
    // setHistory((prevHistory) => [...prevHistory, `$ ${command}`]);
  };

  return (
    <div className="terminal-container">
      <div id="terminal-header">
        <FaTerminal size={12} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
        <span className="terminal-title">Terminal</span>
      </div>
      <div className="terminal-body">
        <div ref={historyRef} className="terminal-history">
          {history.map((line, index) => (
            <div key={index} className="terminal-line">
              {processTerminalLine(line)}
            </div>
          ))}
        </div>
        <div className="terminal-input-line">
          <span className="terminal-prompt">$</span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleCommand(input)}
            className="terminal-input"
            autoFocus
            spellCheck={false}
          />
        </div>
      </div>
    </div>
  );
};