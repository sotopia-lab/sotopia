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
import { FileSystemState, FileNode } from './types/FileSystem';

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
  const [fileSystem, setFileSystem] = useState<FileSystemState>({
    tree: [
      {
        name: 'workspace',
        type: 'folder',
        path: '/workspace',
        children: [
          {
            name: 'index.html',
            type: 'file',
            path: '/workspace/index.html',
          },
          {
            name: 'style.css',
            type: 'file',
            path: '/workspace/style.css',
          },
          {
            name: 'script.js',
            type: 'file',
            path: '/workspace/script.js',
          },
          {
            name: 'main.py',
            type: 'file',
            path: '/workspace/main.py',
          }
        ]
      }
    ],
    files: {
      '/workspace/index.html': '<!DOCTYPE html>\n<html lang="en">...',
      '/workspace/style.css': 'body {\n    background-color: #f0f0f0;...',
      '/workspace/script.js': 'console.log("Hello, World!");',
      '/workspace/main.py': '# Python code here'
    }
  });

  const [currentFile, setCurrentFile] = useState("/workspace/main.py");
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

  const addFile = (path: string, content: string) => {
    setFileSystem(prev => {
      // Split path into parts
      const parts = path.split('/').filter(Boolean);
      const fileName = parts[parts.length - 1];

      // Create new file node
      const newFile: FileNode = {
        name: fileName,
        type: 'file',
        path: path
      };

      // Update tree structure
      const newTree = [...prev.tree];
      let currentLevel = newTree;
      for (let i = 0; i < parts.length - 1; i++) {
        const folder = currentLevel.find(node =>
          node.type === 'folder' && node.name === parts[i]
        );
        if (folder && folder.children) {
          currentLevel = folder.children;
        }
      }

      // Add new file to appropriate level if it doesn't exist
      if (!currentLevel.find(node => node.path === path)) {
        currentLevel.push(newFile);
      }

      // Update files content
      return {
        tree: newTree,
        files: {
          ...prev.files,
          [path]: content
        }
      };
    });

    // Automatically switch to the new/updated file
    setCurrentFile(path);
    setActiveTab('editor');
  };

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

      case "thought":
        setMessages(prev => [...prev, {
          text: `💭 ${agentName} is thinking: ${messageData.data.argument}`,
          type: 'status' as const
        }]);
        break;

      case "write":
        const filePath = messageData.data.path;
        const fileContent = messageData.data.argument;
        addFile(filePath, fileContent);
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
            <FileSystem
              fileSystem={fileSystem.tree}
              onFileSelect={(path) => setCurrentFile(path)}
            />
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
                code={fileSystem.files[currentFile]}
                onChange={(newCode) =>
                  setFileSystem(prevFiles => ({
                    ...prevFiles,
                    files: {
                      ...prevFiles.files,
                      [currentFile]: newCode
                    }
                  }))
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
