import React, { useState } from 'react';
import { ChevronRight, ChevronDown, File } from 'lucide-react';
import {
  SiHtml5, SiCss3, SiJavascript, SiPython,
  SiTypescript, SiJson, SiMarkdown
} from 'react-icons/si';
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

export const FileSystem: React.FC<FileSystemProps> = ({ onFileSelect }) => {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['workspace']));

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

  const renderItem = (item: any, depth: number = 0) => {
    const isExpanded = expandedFolders.has(item.name);

    return (
      <div
        key={item.name}
        className="file-item"
        style={{ paddingLeft: `${depth * 16 + (item.type === 'file' ? 20 : 12)}px` }}
        onClick={() => item.type === 'file' && onFileSelect(`/workspace/${item.name}`)}
      >
        {item.type === 'folder' ? (
          <>
            <span className="folder-arrow" onClick={(e) => toggleFolder(item.name, e)}>
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

  const renderFolder = (folder: any, depth: number = 0) => (
    <div key={folder.name}>
      {renderItem(folder, depth)}
      {expandedFolders.has(folder.name) && folder.children && (
        <div className="folder-children">
          {folder.children.map((child: any) =>
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
      {files.map(file => file.type === 'folder' ?
        renderFolder(file) :
        renderItem(file)
      )}
    </div>
    </>
  );
};
