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
import { useFileSystem } from './hooks/useFileSystems';

// Initialize socket connection to the server
const socket = io('http://localhost:8000', {
  transports: ['websocket'],
  reconnection: true
});

// Log connection status
socket.on('connect', () => {
  console.log('Connected to server with ID:', socket.id);
});

// Log connection errors
socket.on('connect_error', (error) => {
  console.error('Connection error:', error);
});

// Define the type for the active panel in the sidebar
type PanelOption = 'fileSystem' | 'sceneContext';

const App: React.FC = () => {
  // Use custom hook for file system management
  const {
    fileSystem,
    openFiles,
    activeFile,
    handleFileSelect,
    handleFileClose,
    handleFileChange,
    addFile,
    setActiveFile
  } = useFileSystem();

  // State for various components and messages
  const [messages, setMessages] = useState<Array<{text: string, type: 'message' | 'status'}>>([]);
  const [terminalMessages, setTerminalMessages] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'editor' | 'browser'>('editor');
  const [browserUrl, setBrowserUrl] = useState('https://example.com');
  const [activePanel, setActivePanel] = useState<PanelOption>('fileSystem');
  const [sceneMessages, setSceneMessages] = useState<{ text: string, agentName: string }[]>([]);

  // Effect to handle incoming messages from the socket
  useEffect(() => {
    const handleNewMessage = (data: any) => {
      try {
        const messageData = JSON.parse(data.message);

        // Handle Scene context messages
        if (data.channel.startsWith('Scene:')) {
          if (messageData.data?.data_type === "text") {
            setSceneMessages(prev => [...prev, { text: messageData.data.text, agentName: data.channel }]);
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

    // Listen for new messages from the socket
    socket.on('new_message', handleNewMessage);
    return () => {
      // Clean up the listener on component unmount
      socket.off('new_message', handleNewMessage);
    };
  }, []);

  // Function to handle actions from agents
  const handleAgentAction = (messageData: any) => {
    const actionType = messageData.data.action_type;
    const agentName = messageData.data.agent_name;

    console.log('Processing agent action:', actionType, 'from', agentName);

    switch (actionType) {
      case "speak":
        // Handle agent speaking
        const newMessage = {
          text: `${agentName}: ${messageData.data.argument}`,
          type: 'message' as const
        };
        setMessages(prev => [...prev, newMessage]);
        break;

      case "thought":
        // Handle agent's thoughts
        setMessages(prev => [...prev, {
          text: `💭 ${agentName} is thinking: ${messageData.data.argument}`,
          type: 'status' as const
        }]);
        break;

      case "write":
        // Handle file writing
        const filePath = messageData.data.path;
        const fileContent = messageData.data.argument;

        // Check if file already exists in openFiles
        const existingFileIndex = openFiles.findIndex(f => f.path === filePath);

        if (existingFileIndex !== -1) {
          // Update existing file content
          handleFileChange(filePath, fileContent);
        } else {
          // Add new file
          addFile(filePath, fileContent);
        }

        setActiveFile(filePath);
        setActiveTab('editor');
        setActivePanel('fileSystem');
        setMessages(prev => [...prev, {
          text: `${agentName} is writing code...`,
          type: 'status' as const
        }]);
        break;

      case "run":
        // Handle command execution
        setTerminalMessages(prev => [...prev, `$ ${messageData.data.argument}`]);
        setMessages(prev => [...prev, {
          text: `${agentName} is executing a command...`,
          type: 'status' as const
        }]);
        break;

      case "browse":
        // Handle browsing action
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

  // Handle sidebar selection
  const handleSidebarSelect = (option: PanelOption) => {
    setActivePanel(option);
  };

  // Listen for chat messages from the socket
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
