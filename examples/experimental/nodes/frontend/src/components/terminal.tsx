import React, { useState, useRef, useEffect } from 'react';

interface TerminalProps {
  externalMessages: string[];
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
  if (line.startsWith('$')) {
    return <div style={{ color: '#00ff00' }}>{stripAnsiCodes(line)}</div>;
  }
  if (line.includes('notice')) {
    return <div style={{ color: '#808080' }}>{stripAnsiCodes(line)}</div>;
  }
  return <div>{stripAnsiCodes(line)}</div>;
};

export const Terminal: React.FC<TerminalProps> = ({ externalMessages }) => {
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
      setHistory((prevHistory) => [...prevHistory, ...externalMessages]);
    }
  }, [externalMessages]);

  const handleCommand = (command: string) => {
    setHistory([...history, `$ ${command}`, '']);
    setInput('');
  };

  return (
    <div className="terminal-container">
      <div id="terminal-header">
        <span className="terminal-title">Terminal</span>
      </div>
      <div className="terminal-body">
        <div 
          ref={historyRef}
          className="terminal-history"
        >
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