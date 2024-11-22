import React, { useState, useRef, useEffect } from 'react';
import { Socket } from 'socket.io-client';
import './Terminal.css';
import { FaTerminal } from 'react-icons/fa';

interface TerminalProps {
  externalMessages: string[];
  socket: Socket;
}

interface TerminalState {
  user: string;
  hostname: string;
  currentPath: string;
}

interface HistoryEntry {
  prompt: string;
  command?: string;
  output?: string;
}

// Remove ANSI escape codes from terminal output
const stripAnsiCodes = (text: string): string => {
  return text.replace(/\[\d+(?:;\d+)*m|\[\d+m|\[0m|\[1m/g, '');
};

// Process terminal output and apply appropriate styling
const processTerminalLine = (line: string): JSX.Element => {
  const strippedLine = stripAnsiCodes(line);
  
  if (line.startsWith('Requirement already satisfied:')) {
    return <div className="terminal-success">{strippedLine}</div>;
  }
  if (line.includes('notice')) {
    return <div className="terminal-notice">{strippedLine}</div>;
  }
  if (line.includes('error') || line.includes('Error')) {
    return <div className="terminal-error">{strippedLine}</div>;
  }
  return <div className="terminal-text">{strippedLine}</div>;
};

// Normalize file paths by removing duplicate slashes
const normalizePath = (path: string): string => {
  return path.replace(/\/+/g, '/');
};

export const Terminal: React.FC<TerminalProps> = ({ externalMessages, socket }) => {
  // Main terminal state
  const [terminalState, setTerminalState] = useState<TerminalState>({
    user: '',
    hostname: '',
    currentPath: ''
  });

  // Command history
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [input, setInput] = useState('');
  const [isInitialized, setIsInitialized] = useState(false);
  const historyRef = useRef<HTMLDivElement>(null);

  // Initialize terminal on mount
  useEffect(() => {
    if (!isInitialized) {
      socket.emit('terminal_command', 'whoami && hostname && pwd');
      setIsInitialized(true);
    }
  }, [isInitialized, socket]);

  // Auto-scroll to bottom when history updates
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [history]);

  // Handle incoming messages from socket
  useEffect(() => {
    if (externalMessages.length === 0) return;
    
    const message = externalMessages[externalMessages.length - 1];
    if (!message.trim()) return;

    // Handle initialization response
    if (!terminalState.user || !terminalState.hostname || !terminalState.currentPath) {
      const [user, hostname, path] = message.split('\n').map(line => line.trim());
      if (user && hostname && path) {
        setTerminalState({
          user,
          hostname,
          currentPath: path
        });
        return;
      }
    }

    // Handle cd command response
    if (message.startsWith('/')) {
      setTerminalState(prev => ({
        ...prev,
        currentPath: message.trim()
      }));
      return;
    }

    // Update history with command output
    setHistory(prev => {
      const lastEntry = prev[prev.length - 1];
      
      // Skip internal cd command outputs
      if (lastEntry?.command?.startsWith('cd ')) {
        return prev;
      }

      if (lastEntry && !lastEntry.output) {
        // Update last command with its output
        const updatedHistory = [...prev.slice(0, -1)];
        updatedHistory.push({
          ...lastEntry,
          output: message
        });
        return updatedHistory;
      }

      // Add new output entry
      return [...prev, { prompt: getPrompt(), output: message }];
    });
  }, [externalMessages]);

  // Generate terminal prompt string
  const getPrompt = () => {
    const { user, hostname, currentPath } = terminalState;
    if (!user || !hostname) return '$ ';
    return `${user}@${hostname}:${currentPath}$ `;
  };

  // Handle command execution
  const handleCommand = (command: string) => {
    if (!command.trim()) return;
    
    const currentPrompt = getPrompt();
    
    // Add command to history
    setHistory(prev => [...prev, { prompt: currentPrompt, command }]);

    if (command.startsWith('cd ')) {
      const newPath = command.slice(3).trim();
      const targetPath = newPath.startsWith('/')
        ? normalizePath(newPath)
        : newPath === '..'
        ? normalizePath(`${terminalState.currentPath}/..`)
        : normalizePath(`${terminalState.currentPath}/${newPath}`);

      socket.emit('terminal_command', `cd "${targetPath}" && pwd`);
    } else {
      // Execute command in current directory without showing cd
      socket.emit('terminal_command', command);
    }
    
    setInput('');
  };

  // Handle special keys
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleCommand(input);
    }
    // Add more key handlers here (e.g., up/down for history)
  };

  return (
    <div className="terminal-container">
      <div id="terminal-header">
        <FaTerminal size={12} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
        <span className="terminal-title">Terminal</span>
      </div>
      <div className="terminal-body">
        <div ref={historyRef} className="terminal-history">
          {history.map((entry, index) => (
            <div key={index} className="terminal-entry">
              {entry.command && (
                <div className="terminal-command">
                  <span className="terminal-prompt">{entry.prompt}</span>
                  <span>{entry.command}</span>
                </div>
              )}
              {entry.output && (
                <div className="terminal-output">
                  {processTerminalLine(entry.output)}
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="terminal-input-line">
          <span className="terminal-prompt">{getPrompt()}</span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="terminal-input"
            autoFocus
            spellCheck={false}
          />
        </div>
      </div>
    </div>
  );
};