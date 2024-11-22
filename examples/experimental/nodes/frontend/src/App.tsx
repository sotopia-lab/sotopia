import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import './App.css';
import CodeEditor from './components/CodeEditor/CodeEditor';
import { FileSystem } from './components/CodeEditor/FileSystem';
import { Terminal } from './components/Terminal/Terminal';
import { ChatInterface } from './components/ChatInterface/ChatInterface';
import { Browser } from './components/Browser/Browser';
import { Sidebar } from './components/Sidebar/Sidebar';
import { SceneContext } from './components/Sidebar/SceneContext';

const socket = io('http://localhost:8000', {
  transports: ['websocket'],
  reconnection: true
});

socket.on('connect', () => {
  console.log('Connected to server with ID:', socket.id);
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error);
});

type FilesType = {
  [key: string]: string;
};

type PanelOption = 'fileSystem' | 'sceneContext';

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
  const [activePanel, setActivePanel] = useState<PanelOption>('fileSystem');

  useEffect(() => {
    const handleNewMessage = (data: any) => {
      try {
        const messageData = JSON.parse(data.message);
        
        // Check if it's an agent action
        if (messageData.data?.data_type === "agent_action") {
          handleAgentAction(messageData);
        } 
        // Check if it's a command output
        else if (messageData.data?.data_type === "text" && 
                 messageData.data.text.includes("CmdOutputObservation")) {
          const parts = messageData.data.text.split("**CmdOutputObservation (source=None, exit code=0)**");
          if (parts.length > 1) {
            const outputText = parts[1].trim();
            setTerminalMessages(prev => [...prev, outputText]);
          }
        }
        // Log the message for debugging
        console.log('Parsed message:', messageData);
        
      } catch (error) {
        console.error('Error parsing message:', error);
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

    // Always log the action for debugging
    console.log('Processing agent action:', actionType, 'from', agentName);

    switch (actionType) {
      case "speak":
        const newMessage = {
          text: `${agentName}: ${messageData.data.argument}`,
          type: 'message' as const
        };
        setMessages(prev => [...prev, newMessage]);
        break;

      case "write":
        const filePath = messageData.data.path;
        const fileContent = messageData.data.argument;
        setFiles(prev => ({ ...prev, [filePath]: fileContent }));
        setActiveTab('editor');
        setMessages(prev => [...prev, {
          text: `${agentName} is writing code...`,
          type: 'status' as const
        }]);
        break;

      case "run":
        setTerminalMessages(prev => [...prev, `$ ${messageData.data.argument}`]);
        setMessages(prev => [...prev, {
          text: `${agentName} is executing a command...`,
          type: 'status' as const
        }]);
        break;

      case "browse":
        const url = messageData.data.argument;
        setBrowserUrl(url);
        setActiveTab('browser');
        setMessages(prev => [...prev, {
          text: `${agentName} is browsing ${url}`,
          type: 'status' as const
        }]);
        break;

      default:
        console.log('Unknown action type:', actionType);
    }
  };

  const handleSidebarSelect = (option: PanelOption) => {
    setActivePanel(option);
  };

  socket.on('chat_message', (message: string) => {
    setMessages(prev => [...prev, {
      text: message,
      type: 'message' as const
    }]);
  });

  return (
    <div className="App">
      <Sidebar onSelect={handleSidebarSelect} />
      <div id="ide-container">
        {activePanel === 'fileSystem' && (
          <div id="file-explorer">
            <FileSystem onFileSelect={setCurrentFile} />
          </div>
        )}
        {activePanel === 'sceneContext' && (
          <div id="scene-context">
            <SceneContext />
          </div>
        )}
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
          <Terminal externalMessages={terminalMessages} socket={socket}/>
        </div>
      </div>
      <div id="chat-container">
        <ChatInterface 
          messages={messages} 
          socket={socket}
          onSendMessage={(text: string) => {
            setMessages(prev => [...prev, {
              text: `User: ${text}`,
              type: 'message' as const
            }]);
          }}
        />
      </div>
    </div>
  );
};

export default App;
