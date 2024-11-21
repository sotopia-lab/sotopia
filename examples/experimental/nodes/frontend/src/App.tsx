import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import './App.css';
import CodeEditor from './components/code-editor';
import { FileSystem } from './components/file-system';
import { Terminal } from './components/terminal';
import { ChatInterface } from './components/chat-interface';
import { Browser } from './components/browser';

const socket = io();

type FilesType = {
  [key: string]: string;
};

const App: React.FC = () => {
  const [files, setFiles] = useState<FilesType>({
    "/workspace/index.html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>Document</title>\n</head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>",
    "/workspace/style.css": "body {\n    background-color: #f0f0f0;\n    font-family: Arial, sans-serif;\n}",
    "/workspace/script.js": "console.log('Hello, World!');",
    "/workspace/interview.py": "# Python code here"
  });

  const [currentFile, setCurrentFile] = useState("/workspace/interview.py");
  const [messages, setMessages] = useState<Array<{text: string, type: 'message' | 'status'}>>([]);
  const [terminalMessages, setTerminalMessages] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'editor' | 'browser'>('editor');
  const [browserUrl, setBrowserUrl] = useState('https://example.com');

  useEffect(() => {
    const handleNewMessage = (data: any) => {
      const messageData = JSON.parse(data.message);
      if (messageData.data.data_type === "agent_action") {
        handleAgentAction(messageData);
      } else if (messageData.data.data_type === "text" && messageData.data.text.includes("CmdOutputObservation")) {
        const parts = messageData.data.text.split("**CmdOutputObservation (source=None, exit code=0)**");
        if (parts.length > 1) {
          const outputText = parts[1].trim();
          setTerminalMessages((prev) => [...prev, outputText]);
        }
      }
    };

    socket.on('new_message', handleNewMessage);
    return () => {
      socket.off('new_message', handleNewMessage);
    };
  }, []);

  const handleAgentAction = (messageData: any) => {
    const actionType = messageData.data.action_type;
    const agentName = messageData.data.agent_name;
    
    if (actionType === "speak") {
      const newMessage = {
        text: `${agentName}: ${messageData.data.argument}`,
        type: 'message' as const
      };
      setMessages(prev => [...prev, newMessage]);
    } else if (actionType === "write") {
      const filePath = messageData.data.path;
      const fileContent = messageData.data.argument;
      setFiles(prevFiles => ({ ...prevFiles, [filePath]: fileContent }));
      setActiveTab('editor');
      const statusMessage = {
        text: `${agentName} is writing code...`,
        type: 'status' as const
      };
      setMessages(prev => [...prev, statusMessage]);
    } else if (actionType === "run") {
      setTerminalMessages((prev) => [...prev, `$ ${messageData.data.argument}`]);
      const statusMessage = {
        text: `${agentName} is executing a command...`,
        type: 'status' as const
      };
      setMessages(prev => [...prev, statusMessage]);
    } else if (actionType === "browse") {
      const url = messageData.data.argument;
      setBrowserUrl(url);
      setActiveTab('browser');
      const statusMessage = {
        text: `${agentName} is browsing ${url}`,
        type: 'status' as const
      };
      setMessages(prev => [...prev, statusMessage]);
    }
  };

  return (
    <div className="App">
      <div id="ide-container">
        <div id="file-explorer">
          <FileSystem onFileSelect={setCurrentFile} />
        </div>
        <div id="code-interface">
          <div className="tabs">
            <button 
              onClick={() => setActiveTab('editor')} 
              className={activeTab === 'editor' ? 'active' : ''}
            >
              Code Editor
            </button>
            <button 
              onClick={() => setActiveTab('browser')} 
              className={activeTab === 'browser' ? 'active' : ''}
            >
              Browser
            </button>
          </div>
          {activeTab === 'editor' ? (
            <div id="code-editor">
              <CodeEditor
                code={files[currentFile]}
                onChange={(newCode) => 
                  setFiles(prevFiles => ({ ...prevFiles, [currentFile]: newCode }))
                }
                filename={currentFile}
              />
            </div>
          ) : (
            <Browser url={browserUrl} />
          )}
          <Terminal externalMessages={terminalMessages} />
        </div>
      </div>
      <div id="chat-container">
        <ChatInterface messages={messages} />
      </div>
    </div>
  );
};

export default App;