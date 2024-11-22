import React from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { html } from '@codemirror/lang-html';
import { css } from '@codemirror/lang-css';
import { python } from '@codemirror/lang-python';
import { githubDark } from '@uiw/codemirror-theme-github';
import { EditorView } from '@codemirror/view';
import './CodeEditor.css'; // Import the CSS file
import { X } from 'lucide-react';
import { File } from 'lucide-react';
import { SiHtml5, SiCss3, SiJavascript, SiPython, SiTypescript } from 'react-icons/si'; // Import icons

interface OpenFile {
  path: string;
  content: string;
}

interface CodeEditorProps {
  openFiles: OpenFile[];
  activeFile?: string;
  onFileClose: (path: string) => void;
  onFileSelect: (path: string) => void;
  onChange: (path: string, content: string) => void;
}

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

const CodeEditor: React.FC<CodeEditorProps> = ({
  openFiles,
  activeFile,
  onFileClose,
  onFileSelect,
  onChange,
}) => {
  const getFileName = (path: string) => path.split('/').pop() || path;

  const handleClose = (e: React.MouseEvent, path: string) => {
    e.stopPropagation();
    onFileClose(path);
  };

  const activeFileContent = openFiles.find(f => f.path === activeFile)?.content;

  // Add empty lines to fill the editor
  const fillEmptyLines = (content: string | undefined) => {
    if (!content) return '\n'.repeat(50); // Return 50 empty lines if no content

    const lines = content.split('\n');
    const currentLines = lines.length;
    if (currentLines < 50) {
      return content + '\n'.repeat(50 - currentLines);
    }
    return content;
  };

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
        return [javascript()];
    }
  };

  return (
    <div className="editor-container">
      <div className="editor-tabs">
        {openFiles.map((file) => (
          <div
            key={file.path}
            className={`editor-tab ${file.path === activeFile ? 'active' : ''}`}
            onClick={() => onFileSelect(file.path)}
          >
            {getFileIcon(file.path)}
            <span className="tab-title">{getFileName(file.path)}</span>
            <span
              className="tab-close"
              onClick={(e) => handleClose(e, file.path)}
            >
              <X size={14} />
            </span>
          </div>
        ))}
      </div>
      {activeFile && (
        <div className="editor-content">
          <CodeMirror
            value={fillEmptyLines(activeFileContent)}
            height="100%"
            theme={githubDark}
            extensions={[
              ...getLanguageExtension(activeFile),
              EditorView.lineWrapping,
            ]}
            onChange={(value) => onChange(activeFile, value)}
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
