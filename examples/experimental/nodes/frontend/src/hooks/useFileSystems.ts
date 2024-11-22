/**
 * useFileSystems.ts
 *
 * This custom hook manages the state of a file system within the application. It provides
 * functionality for handling file selection, opening, closing, and editing files. The hook
 * maintains the structure of the file system and the currently open files, allowing users
 * to interact with files in a simulated environment.
 *
 * Key Features:
 * - Initializes a file system with a predefined structure and files.
 * - Allows users to open, close, and edit files.
 * - Updates the file system state when files are modified or added.
 *
 * Returns:
 * - fileSystem: The current state of the file system.
 * - openFiles: An array of currently open files.
 * - activeFile: The path of the currently active file.
 * - handleFileSelect: Function to select a file to open.
 * - handleFileClose: Function to close an open file.
 * - handleFileChange: Function to update the content of an open file.
 * - addFile: Function to add a new file to the file system.
 * - setActiveFile: Function to set the currently active file.
 *
 */

import { useState } from 'react';
import { FileSystemState, FileNode, OpenFile } from '../types/FileSystem';

// Initial file system structure and files
const initialFileSystem: FileSystemState = {
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
};

// Custom hook for managing the file system
export const useFileSystem = () => {
  const [fileSystem, setFileSystem] = useState<FileSystemState>(initialFileSystem); // State for the file system
  const [openFiles, setOpenFiles] = useState<OpenFile[]>([
    {
      path: '/workspace/main.py',
      content: initialFileSystem.files['/workspace/main.py']
    }
  ]);
  const [activeFile, setActiveFile] = useState<string>('/workspace/main.py'); // State for the active file

  // Function to handle file selection
  const handleFileSelect = (path: string) => {
    if (!openFiles.find(f => f.path === path)) {
      setOpenFiles(prev => [...prev, {
        path,
        content: fileSystem.files[path] || ''
      }]);
    }
    setActiveFile(path); // Set the selected file as active
  };

  // Function to handle closing a file
  const handleFileClose = (path: string) => {
    setOpenFiles(prev => prev.filter(f => f.path !== path)); // Remove the file from open files
    if (activeFile === path) {
      const remainingFiles = openFiles.filter(f => f.path !== path);
      setActiveFile(remainingFiles[remainingFiles.length - 1]?.path); // Set the last opened file as active
    }
  };

  // Function to handle changes in file content
  const handleFileChange = (path: string, content: string) => {
    setOpenFiles(prev =>
      prev.map(f => f.path === path ? { ...f, content } : f) // Update the content of the file
    );

    setFileSystem(prev => ({
      ...prev,
      files: {
        ...prev.files,
        [path]: content // Update the file system state with new content
      }
    }));
  };

  // Function to add a new file to the file system
  const addFile = (path: string, content: string) => {
    setFileSystem(prev => {
      const newFiles = {
        ...prev.files,
        [path]: content // Add the new file to the files object
      };

      const pathParts = path.split('/').filter(Boolean);
      const fileName = pathParts[pathParts.length - 1];

      const updatedTree = [...prev.tree];
      const workspaceFolder = updatedTree.find(node => node.name === 'workspace');
      if (workspaceFolder && workspaceFolder.children) {
        const fileExists = workspaceFolder.children.some(child => child.path === path);

        if (!fileExists) {
          const newFileNode: FileNode = {
            name: fileName,
            type: 'file',
            path: path,
          };
          workspaceFolder.children = [...workspaceFolder.children, newFileNode]; // Add the new file node to the workspace
        }
      }

      return {
        tree: updatedTree,
        files: newFiles // Return the updated file system state
      };
    });

    setOpenFiles(prev => {
      const existingFileIndex = prev.findIndex(f => f.path === path);
      if (existingFileIndex !== -1) {
        const updatedFiles = [...prev];
        updatedFiles[existingFileIndex] = { path, content }; // Update existing file content
        return updatedFiles;
      }
      return [...prev, { path, content }]; // Add the new file to open files
    });
  };

  return {
    fileSystem,
    openFiles,
    activeFile,
    handleFileSelect,
    handleFileClose,
    handleFileChange,
    addFile,
    setActiveFile
  };
};
