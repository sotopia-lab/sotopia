import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { html } from '@codemirror/lang-html';
import { css } from '@codemirror/lang-css';
import { python } from '@codemirror/lang-python';
import { githubDark } from '@uiw/codemirror-theme-github';
import './CodeEditor.css'; // Import the CSS file

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
  filename: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ code, onChange, filename }) => {
  // Add empty lines to fill the editor
  const fillEmptyLines = (content: string) => {
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
      <div className="editor-header">
        {filename}
      </div>
      <div className="editor-content">
        <CodeMirror
          value={fillEmptyLines(code)}
          height="100%"
          theme={githubDark}
          extensions={getLanguageExtension(filename)}
          onChange={onChange}
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
    </div>
  );
};

export default CodeEditor;
