/**
 * FileSystem.tsx
 *
 * This component represents a file system explorer within the application. It allows users to
 * navigate through folders and files, selecting files to open in the code editor. The file
 * system is displayed in a tree structure, with expandable folders and icons representing
 * different file types.
 *
 * Key Features:
 * - Displays a hierarchical view of folders and files.
 * - Supports expanding and collapsing folders to show/hide their contents.
 * - Uses icons to represent different file types (HTML, CSS, JavaScript, Python, etc.).
 * - Allows users to select files, triggering a callback to open them in the code editor.
 *
 * Props:
 * - fileSystem: An array of FileNode objects representing the file system structure.
 * - onFileSelect: A callback function to handle the selection of a file.
 * - socket: A socket object to handle communication with the backend.
 *
 */

import React, { useState } from 'react';
import { ChevronRight, ChevronDown, File, RefreshCw } from 'lucide-react';
import {
  SiHtml5, SiCss3, SiJavascript, SiPython,
  SiTypescript, SiJson, SiMarkdown
} from 'react-icons/si';
import './FileSystem.css'; // Import the CSS file
import { FileNode } from '../../types/FileSystem'; // Import the FileNode type
import { Socket } from 'socket.io-client';

// Define the props for the FileSystem component
interface FileSystemProps {
  fileSystem: FileNode[];
  onFileSelect: (path: string) => void;
  socket: Socket; // Add socket prop
}

// Function to get the appropriate file icon based on the file extension
const getFileIcon = (fileName: string) => {
  const ext = fileName.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'html': return <SiHtml5 size={10} className="file-icon html-icon" />;
    case 'css': return <SiCss3 size={10} className="file-icon css-icon" />;
    case 'js': return <SiJavascript size={10} className="file-icon js-icon" />;
    case 'py': return <SiPython size={10} className="file-icon python-icon" />;
    case 'ts':
    case 'tsx': return <SiTypescript size={10} className="file-icon ts-icon" />;
    case 'json': return <SiJson size={10} className="file-icon json-icon" />;
    case 'md': return <SiMarkdown size={10} className="file-icon md-icon" />;
    default: return <File size={10} className="file-icon" />;
  }
};

// Main FileSystem component definition
export const FileSystem: React.FC<FileSystemProps> = ({ fileSystem, onFileSelect, socket }) => {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['/workspace'])); // Track expanded folders

  const handleRefresh = () => {
    // Send command to get file structure
    socket.emit('terminal_command', "find /workspace -type f");
  };

  // Toggle the expansion state of a folder
  const toggleFolder = (folderName: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent event bubbling
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folderName)) {
        next.delete(folderName); // Collapse the folder
      } else {
        next.add(folderName); // Expand the folder
      }
      return next;
    });
  };

  // Render a file or folder item
  const renderItem = (item: FileNode, depth: number = 0) => {
    const isExpanded = expandedFolders.has(item.path); // Check if the folder is expanded

    return (
      <div
        key={item.path}
        className="file-item"
        style={{ paddingLeft: `${depth * 16 + (item.type === 'file' ? 20 : 12)}px` }} // Indent based on depth
        onClick={() => item.type === 'file' && onFileSelect(item.path)} // Select file on click
      >
        {item.type === 'folder' ? (
          <>
            <span className="folder-arrow" onClick={(e) => toggleFolder(item.path, e)}>
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </span>
          </>
        ) : (
          getFileIcon(item.name) // Display the file icon
        )}
        <span>{item.name}</span> {/* Display the file or folder name */}
      </div>
    );
  };

  // Render a folder and its children
  const renderFolder = (folder: FileNode, depth: number = 0) => (
    <div key={folder.path}>
      {renderItem(folder, depth)}
      {expandedFolders.has(folder.path) && folder.children && (
        <div className="folder-children">
          {folder.children.map((child: FileNode) =>
            child.type === 'folder' ?
              renderFolder(child, depth + 1) : // Recursively render folders
              renderItem(child, depth + 1) // Render files
          )}
        </div>
      )}
    </div>
  );

  return (
    <>
      <div id="file-explorer-header">
        Folders
        <button
          onClick={handleRefresh}
          className="refresh-button"
          title="Refresh file structure"
        >
          <RefreshCw size={14} />
        </button>
      </div>
      <div className="file-explorer">
        {fileSystem.map(node =>
          node.type === 'folder' ? renderFolder(node) : renderItem(node) // Render the file system
        )}
      </div>
    </>
  );
};
