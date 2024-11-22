import { useState } from 'react';
import { FileSystemState, FileNode, OpenFile } from '../types/FileSystem';

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

export const useFileSystem = () => {
  const [fileSystem, setFileSystem] = useState<FileSystemState>(initialFileSystem);
  const [openFiles, setOpenFiles] = useState<OpenFile[]>([
    {
      path: '/workspace/main.py',
      content: initialFileSystem.files['/workspace/main.py']
    }
  ]);
  const [activeFile, setActiveFile] = useState<string>('/workspace/main.py');

  const handleFileSelect = (path: string) => {
    if (!openFiles.find(f => f.path === path)) {
      setOpenFiles(prev => [...prev, {
        path,
        content: fileSystem.files[path] || ''
      }]);
    }
    setActiveFile(path);
  };

  const handleFileClose = (path: string) => {
    setOpenFiles(prev => prev.filter(f => f.path !== path));
    if (activeFile === path) {
      const remainingFiles = openFiles.filter(f => f.path !== path);
      setActiveFile(remainingFiles[remainingFiles.length - 1]?.path);
    }
  };

  const handleFileChange = (path: string, content: string) => {
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
    setFileSystem(prev => {
      const newFiles = {
        ...prev.files,
        [path]: content
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
          workspaceFolder.children = [...workspaceFolder.children, newFileNode];
        }
      }

      return {
        tree: updatedTree,
        files: newFiles
      };
    });

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
