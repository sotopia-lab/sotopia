import React, { useState } from 'react';
import { ChevronRight, ChevronDown, File } from 'lucide-react';
import {
  SiHtml5, SiCss3, SiJavascript, SiPython,
  SiTypescript, SiJson, SiMarkdown
} from 'react-icons/si';
import './FileSystem.css'; // Import the CSS file
import { FileNode } from '../../types/FileSystem';

interface FileSystemProps {
  fileSystem: FileNode[];
  onFileSelect: (path: string) => void;
}

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

export const FileSystem: React.FC<FileSystemProps> = ({ fileSystem, onFileSelect }) => {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['/workspace']));

  const toggleFolder = (folderName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folderName)) {
        next.delete(folderName);
      } else {
        next.add(folderName);
      }
      return next;
    });
  };

  const renderItem = (item: FileNode, depth: number = 0) => {
    const isExpanded = expandedFolders.has(item.path);

    return (
      <div
        key={item.path}
        className="file-item"
        style={{ paddingLeft: `${depth * 16 + (item.type === 'file' ? 20 : 12)}px` }}
        onClick={() => item.type === 'file' && onFileSelect(item.path)}
      >
        {item.type === 'folder' ? (
          <>
            <span className="folder-arrow" onClick={(e) => toggleFolder(item.path, e)}>
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </span>
          </>
        ) : (
          getFileIcon(item.name)
        )}
        <span>{item.name}</span>
      </div>
    );
  };

  const renderFolder = (folder: FileNode, depth: number = 0) => (
    <div key={folder.path}>
      {renderItem(folder, depth)}
      {expandedFolders.has(folder.path) && folder.children && (
        <div className="folder-children">
          {folder.children.map((child: FileNode) =>
            child.type === 'folder' ?
              renderFolder(child, depth + 1) :
              renderItem(child, depth + 1)
          )}
        </div>
      )}
    </div>
  );

  return (
    <>
      <div id="file-explorer-header">Folders</div>
      <div className="file-explorer">
        {fileSystem.map(node =>
          node.type === 'folder' ? renderFolder(node) : renderItem(node)
        )}
      </div>
    </>
  );
};
