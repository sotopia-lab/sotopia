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
 * - updateFileSystemFromList: Function to update the file system state when receiving the file list from the server.
 *
 */
"use client"

import { useState } from 'react';
import { FileSystemState, FileNode, OpenFile } from '../types/FileSystem';

// Initial file system structure and files
const initialFileSystem: FileSystemState = {
  tree: [
    {
      name: 'workspace',
      type: 'folder',
      path: '/workspace',
      children: []
    }
  ],
  files: {
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
  const [activeFile, setActiveFile] = useState<string>(); // State for the active file

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

  const updateFileSystemFromList = (fileList: string[]) => {
    // Initialize root workspace folder
    const newTree: FileNode[] = [{
      name: 'workspace',
      type: 'folder',
      path: '/workspace',
      children: []
    }];

    // Create a set of existing file paths for quick lookup
    const existingFiles = new Set(Object.keys(fileSystem.files));

    // Process each file path
    fileList.forEach(filePath => {
      // Clean up the file path
      const cleanedPath = filePath.replace(/^\/workspace\//, '').trim().replace(/\\r/g, ''); // Remove \r
      const segments = cleanedPath.split('/').filter(Boolean);

      if (segments.length === 0) return; // Skip if it's just the workspace folder

      let currentLevel = newTree[0].children!;
      let currentPath = '/workspace';

      // Process each segment of the path
      segments.forEach((segment, index) => {
        currentPath += '/' + segment;

        // If we're at the last segment, it's a file
        if (index === segments.length - 1) {
          // Check if the file already exists
          if (!existingFiles.has(currentPath)) {
            currentLevel.push({
              name: segment,
              type: 'file',
              path: currentPath
            });
          }
        } else {
          // It's a folder
          let folder = currentLevel.find(
            node => node.type === 'folder' && node.name === segment
          );

          // Create folder if it doesn't exist
          if (!folder) {
            folder = {
              name: segment,
              type: 'folder',
              path: currentPath,
              children: []
            };
            currentLevel.push(folder);
          }

          currentLevel = folder.children!;
        }
      });
    });

    // Sort folders and files
    const sortNodes = (nodes: FileNode[]) => {
      return nodes.sort((a, b) => {
        // Folders come before files
        if (a.type !== b.type) {
          return a.type === 'folder' ? -1 : 1;
        }
        // Alphabetical sorting within same type
        return a.name.localeCompare(b.name);
      });
    };

    // Recursively sort all levels
    const sortRecursive = (node: FileNode) => {
      if (node.children) {
        node.children = sortNodes(node.children);
        node.children.forEach(child => {
          if (child.type === 'folder') {
            sortRecursive(child);
          }
        });
      }
    };

    // Sort the tree
    sortRecursive(newTree[0]);

    // Update the file system state while preserving existing file contents
    setFileSystem(prev => ({
      ...prev,
      tree: newTree
    }));
  };

  return {
    fileSystem,
    openFiles,
    activeFile,
    handleFileSelect,
    handleFileClose,
    handleFileChange,
    addFile,
    setActiveFile,
    updateFileSystemFromList
  };
};
