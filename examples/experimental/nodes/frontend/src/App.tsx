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

type OpenFile = {
  path: string;
  content: string;
};

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

  const [openFiles, setOpenFiles] = useState<OpenFile[]>([
    {
      path: '/workspace/main.py',
      content: fileSystem.files['/workspace/main.py']
    }
  ]);
  const [activeFile, setActiveFile] = useState<string>('/workspace/main.py');
  const [messages, setMessages] = useState<Array<{text: string, type: 'message' | 'status'}>>([]);
  const [terminalMessages, setTerminalMessages] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'editor' | 'browser'>('editor');
  const [browserUrl, setBrowserUrl] = useState('https://example.com');
  const [activePanel, setActivePanel] = useState<PanelOption>('fileSystem');
  const [sceneMessages, setSceneMessages] = useState<{ text: string, agentName: string }[]>([]);

  useEffect(() => {
    const handleNewMessage = (data: any) => {
      try {
        const messageData = JSON.parse(data.message);

        // Handle Scene context messages
        if (data.channel.startsWith('Scene:')) {
          if (messageData.data?.data_type === "text") {
            setSceneMessages(prev => [...prev, { text: messageData.data.text, agentName: data.channel }]); // Changed channelName to agentName
            setActivePanel('sceneContext');
          }
          return;
        }

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

  const handleFileSelect = (path: string) => {
    // Check if file is already open
    if (!openFiles.find(f => f.path === path)) {
      // Add new file to openFiles
      setOpenFiles(prev => [...prev, {
        path,
        content: fileSystem.files[path] || ''
      }]);
    }
    setActiveFile(path);
    setActiveTab('editor');
  };

  const handleFileClose = (path: string) => {
    setOpenFiles(prev => prev.filter(f => f.path !== path));
    if (activeFile === path) {
      // Set the active file to the last remaining open file
      const remainingFiles = openFiles.filter(f => f.path !== path);
      setActiveFile(remainingFiles[remainingFiles.length - 1]?.path);
    }
  };

  const handleFileChange = (path: string, content: string) => {
    // Update both openFiles and fileSystem
    setOpenFiles(prev =>
      prev.map(f => f.path === path ? { ...f, content } : f)
    );

    setFileSystem(prev => ({
      ...prev,
      files: {
        ...prev.files,
        [path]: content
      }
    }));
  };

  const addFile = (path: string, content: string) => {
    // Update fileSystem
    setFileSystem(prev => {
      const newFiles = {
        ...prev.files,
        [path]: content
      };
      
      // Create new file node
      const pathParts = path.split('/').filter(Boolean);
      const fileName = pathParts[pathParts.length - 1];
      const newFileNode: FileNode = {
        name: fileName,
        type: 'file',
        path: path,
      };
      
      // Add file node to tree
      const updatedTree = [...prev.tree];
      const workspaceFolder = updatedTree.find(node => node.name === 'workspace');
      if (workspaceFolder && workspaceFolder.children) {
        workspaceFolder.children = [...workspaceFolder.children, newFileNode];
      }

      return {
        tree: updatedTree,
        files: newFiles
      };
    });

    // Update openFiles
    setOpenFiles(prev => {
      const existingFileIndex = prev.findIndex(f => f.path === path);
      if (existingFileIndex !== -1) {
        const updatedFiles = [...prev];
        updatedFiles[existingFileIndex] = { path, content };
        return updatedFiles;
      }
      return [...prev, { path, content }];
    });
  };

  const handleAgentAction = (messageData: any) => {
    const actionType = messageData.data.action_type;
    const agentName = messageData.data.agent_name;

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
        setActiveFile(filePath); // This will now activate the existing tab if it exists
        setActiveTab('editor');
        setActivePanel('fileSystem');
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
              onFileSelect={handleFileSelect}
            />
          </div>
        )}
        {activePanel === 'sceneContext' && (
          <div id="scene-context">
            <SceneContext messages={sceneMessages} />
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
                openFiles={openFiles}
                activeFile={activeFile}
                onFileClose={handleFileClose}
                onFileSelect={setActiveFile}
                onChange={handleFileChange}
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
