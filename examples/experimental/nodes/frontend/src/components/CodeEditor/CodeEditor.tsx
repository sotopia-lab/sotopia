/**
 * CodeEditor.tsx
 *
 * This component represents a code editor interface within the application. It allows users to
 * open, edit, and manage multiple code files simultaneously. The code editor uses CodeMirror
 * for syntax highlighting and code editing features. It communicates with the parent component
 * to handle file selection, closing, and content changes.
 *
 * Key Features:
 * - Displays tabs for each open file with icons based on file type.
 * - Supports syntax highlighting for various programming languages (JavaScript, HTML, CSS, Python).
 * - Allows users to close files and switch between them.
 * - Automatically fills empty lines to maintain a consistent editor height.
 *
 * Props:
 * - openFiles: An array of objects representing the currently open files, each containing a path and content.
 * - activeFile: The path of the currently active file being edited.
 * - onFileClose: A callback function to handle the closing of a file.
 * - onFileSelect: A callback function to handle the selection of a file.
 * - onChange: A callback function to handle changes in the content of the active file.
 *
 */
"use client"

import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { html } from '@codemirror/lang-html';
import { css } from '@codemirror/lang-css';
import { python } from '@codemirror/lang-python';
import { githubDark } from '@uiw/codemirror-theme-github';
import { EditorView } from '@codemirror/view';
import '../../styles/globals.css'; // Import the CSS file
import { X } from 'lucide-react';
import { File } from 'lucide-react';
import { SiHtml5, SiCss3, SiJavascript, SiPython, SiTypescript } from 'react-icons/si'; // Import icons
import { Save } from 'lucide-react'; // Import save icon
import { Socket } from 'socket.io-client';

// Define the structure of an open file
interface OpenFile {
  path: string;
  content: string;
}

// Define the props for the CodeEditor component
interface CodeEditorProps {
  openFiles: OpenFile[];
  activeFile?: string;
  onFileClose: (path: string) => void;
  onFileSelect: (path: string) => void;
  onChange: (path: string, content: string) => void;
  socket: Socket;
}

// Function to get the appropriate file icon based on the file extension
const getFileIcon = (path: string) => {
  const ext = path.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'html': return <SiHtml5 size={10} className="file-icon html-icon" />;
    case 'css': return <SiCss3 size={10} className="file-icon css-icon" />;
    case 'js': return <SiJavascript size={10} className="file-icon js-icon" />;
    case 'py': return <SiPython size={10} className="file-icon python-icon" />;
    case 'ts':
    case 'tsx': return <SiTypescript size={10} className="file-icon ts-icon" />;
    default: return <File size={10} className="file-icon" />;
  }
};

// Main CodeEditor component definition
const CodeEditor: React.FC<CodeEditorProps> = ({
  openFiles,
  activeFile,
  socket,
  onFileClose,
  onFileSelect,
  onChange,
}) => {
  // Function to extract the file name from the file path
  const getFileName = (path: string) => path.split('/').pop() || path;
  const [unsavedChanges, setUnsavedChanges] = useState<{ [key: string]: boolean }>({}); // Track unsaved changes for each file
  const activeFileContent = openFiles.find(f => f.path === activeFile)?.content;   // Get the content of the currently active file

  // Handle file close action
  const handleClose = (e: React.MouseEvent, path: string) => {
    e.stopPropagation();
    onFileClose(path);
  };

  // Add empty lines to fill the editor to a minimum height
  const fillEmptyLines = (content: string | undefined) => {
    if (!content) return '\n'.repeat(50); // Return 50 empty lines if no content

    const lines = content.split('\n');
    const currentLines = lines.length;
    if (currentLines < 50) {
      return content + '\n'.repeat(50 - currentLines);
    }
    return content;
  };

  // Determine the language extension for syntax highlighting
  const getLanguageExtension = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'js':
        return [javascript({ jsx: true })];
      case 'html':
        return [html()];
      case 'css':
        return [css()];
      case 'py':
        return [python()];
      default:
        return [javascript()]; // Default to JavaScript
    }
  };

  const handleSave = () => {
    if (activeFile) {
      console.log('Saving file:', activeFile, activeFileContent);
      socket.emit('save_file', { path: activeFile, content: activeFileContent });
      setUnsavedChanges(prev => ({ ...prev, [activeFile]: false })); // Reset unsaved changes for the active file
    }
  };

  const handleChange = (activeFile: string, value: string) => {
    onChange(activeFile, value); // Call the onChange prop
    setUnsavedChanges(prev => ({ ...prev, [activeFile]: true })); // Set unsaved changes for the active file
  };

  return (
    <div className="editor-container">
      <div className="editor-toolbar">
        <div className="editor-tabs">
          {openFiles.map((file) => (
            <div
              key={file.path}
              className={`editor-tab ${file.path === activeFile ? 'active' : ''}`}
              onClick={() => onFileSelect(file.path)}
            >
              {getFileIcon(file.path)}
              <span className="tab-title">
              {unsavedChanges[file.path] ? `* ${getFileName(file.path)}` : getFileName(file.path)}
              </span>
              <span
                className="tab-close"
                onClick={(e) => handleClose(e, file.path)}
              >
                <X size={14} />
              </span>
            </div>
          ))}

          <button onClick={handleSave} className="save-button">
            <Save size={16} /> {/* Save icon */}
          </button>
        </div>
      </div>
      {activeFile && (
        <div className="editor-content">
          <CodeMirror
            value={fillEmptyLines(activeFileContent)} // Fill empty lines if needed
            height="100%"
            theme={githubDark} // Set the theme for the editor
            extensions={[
              ...getLanguageExtension(activeFile), // Set language-specific extensions
              EditorView.lineWrapping, // Enable line wrapping
            ]}
            onChange={(value) => handleChange(activeFile, value)} // Handle content changes
            basicSetup={{
              lineNumbers: true,
              highlightActiveLineGutter: true,
              highlightSpecialChars: true,
              foldGutter: true,
              drawSelection: true,
              dropCursor: true,
              allowMultipleSelections: true,
              indentOnInput: true,
              bracketMatching: true,
              closeBrackets: true,
              autocompletion: true,
              rectangularSelection: true,
              crosshairCursor: true,
              highlightActiveLine: true,
              highlightSelectionMatches: true,
              closeBracketsKeymap: true,
              defaultKeymap: true,
              searchKeymap: true,
              historyKeymap: true,
              foldKeymap: true,
              completionKeymap: true,
              lintKeymap: true
            }}
          />
        </div>
      )}
    </div>
  );
};

export default CodeEditor;
