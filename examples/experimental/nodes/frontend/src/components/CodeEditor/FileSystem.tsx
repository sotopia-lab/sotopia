import React from 'react';
import { Folder, File } from 'lucide-react';
import './FileSystem.css'; // Import the CSS file

const files = [
  { name: 'workspace', type: 'folder', children: [
    { name: 'index.html', type: 'file' },
    { name: 'style.css', type: 'file' },
    { name: 'script.js', type: 'file' },
    { name: 'interview.py', type: 'file' },
  ]},
];

interface FileSystemProps {
  onFileSelect: (fileName: string) => void;
}

export const FileSystem: React.FC<FileSystemProps> = ({ onFileSelect }) => {
  const renderItem = (item: any, depth: number = 0) => (
    <div
      key={item.name}
      className="file-item"
      style={{ paddingLeft: `${depth * 12 + 12}px` }}
      onClick={() => item.type === 'file' && onFileSelect(`/workspace/${item.name}`)}
    >
      {item.type === 'folder' ?
        <Folder size={16} className="folder-icon" /> :
        <File size={16} className="file-icon" />
      }
      <span>{item.name}</span>
    </div>
  );

  const renderFolder = (folder: any, depth: number = 0) => (
    <div key={folder.name}>
      {renderItem(folder, depth)}
      {folder.children && folder.children.map((child: any) =>
        child.type === 'folder' ?
          renderFolder(child, depth + 1) :
          renderItem(child, depth + 1)
      )}
    </div>
  );

  return (
    <>
    <div id="file-explorer-header">Folders</div>
    <div className="file-explorer">
      {files.map(file => file.type === 'folder' ?
        renderFolder(file) :
        renderItem(file)
      )}
    </div>
    </>
  );
};
