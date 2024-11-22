export interface FileNode {
  name: string;
  type: 'file' | 'folder';
  path: string;
  content?: string;
  children?: FileNode[];
}

export interface FileSystemState {
  tree: FileNode[];
  files: {
    [path: string]: string;  // path -> content mapping
  };
}

export interface OpenFile {
  path: string;
  content: string;
}

export interface CodeEditorState {
  openFiles: OpenFile[];
  activeFile?: string; // path of the currently active file
}
