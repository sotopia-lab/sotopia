'use client';

import { useEffect, useState } from 'react';
import io, { Socket } from 'socket.io-client';
import CodeEditor from '@/components/CodeEditor/CodeEditor';
import { FileSystem } from '@/components/CodeEditor/FileSystem';
import { Terminal } from '@/components/Terminal/Terminal';
import { ChatInterface } from '@/components/ChatInterface/ChatInterface';
import { Browser } from '@/components/Browser/Browser';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { SceneContext } from '@/components/Sidebar/SceneContext';

type FilesType = {
  [key: string]: string;
};

type PanelOption = 'fileSystem' | 'sceneContext';

let socket: Socket;

export default function Home() {
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
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    socket = io(process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:8000', {
      transports: ['websocket'],
      reconnection: true
    });

    socket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to server with ID:', socket.id);
    });

    socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setIsConnected(false);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!socket) return;

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
  }, [socket]);

  const handleAgentAction = (messageData: any) => {
    const actionType = messageData.data.action_type;
    const agentName = messageData.data.agent_name;

    switch (actionType) {
      case "speak":
        setMessages(prev => [...prev, {
          text: `${agentName}: ${messageData.data.argument}`,
          type: 'message' as const
        }]);
        break;
      case "write":
        const filePath = messageData.data.path;
        const fileContent = messageData.data.argument;
        setFiles(prevFiles => ({ ...prevFiles, [filePath]: fileContent }));
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
    }
  };

  if (!isConnected) {
    return <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
      Connecting to server...
    </div>;
  }

  return (
    <div className="flex h-screen bg-gray-900">
      <Sidebar onSelect={setActivePanel} />
      <div className="flex flex-1">
        {activePanel === 'fileSystem' && (
          <div id="file-explorer" className="w-64 bg-gray-800 border-r border-gray-700">
            <FileSystem onFileSelect={setCurrentFile} />
          </div>
        )}
        {activePanel === 'sceneContext' && (
          <div id="scene-context" className="w-64 bg-gray-800 border-r border-gray-700">
            <SceneContext />
          </div>
        )}
        <div className="flex-1 flex flex-col">
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setActiveTab('editor')}
              className={`px-4 py-2 ${activeTab === 'editor' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800'}`}
            >
              Code Editor
            </button>
            <button
              onClick={() => setActiveTab('browser')}
              className={`px-4 py-2 ${activeTab === 'browser' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800'}`}
            >
              Browser
            </button>
          </div>
          <div className="flex-1">
            {activeTab === 'editor' ? (
              <CodeEditor
                code={files[currentFile]}
                onChange={(newCode) => setFiles(prev => ({ ...prev, [currentFile]: newCode }))}
                filename={currentFile}
              />
            ) : (
              <Browser url={browserUrl} />
            )}
          </div>
          <Terminal externalMessages={terminalMessages} socket={socket} />
        </div>
        <div className="w-[600px] border-l border-gray-700">
          <ChatInterface messages={messages} socket={socket} />
        </div>
      </div>
    </div>
  );
}
