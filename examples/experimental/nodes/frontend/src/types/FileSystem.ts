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
