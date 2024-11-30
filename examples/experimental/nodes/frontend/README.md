# AI Code Editor Frontend

A modern, interactive code editor interface designed to work with AI agents. This React-based application provides real-time code editing, file management, and chat capabilities.

## Core Features

### 1. Code Editor Interface
- Multi-file editing support with tabs
- Syntax highlighting for multiple languages
- Real-time code updates
- File system navigation

### 2. Chat Interface
- Real-time communication with AI agents
- Status messages for agent actions
- Visual indicators for new messages
- Avatar-based message display

### 3. Terminal Integration
- Command execution display
- Output visualization
- Real-time updates

### 4. File System Management
- Tree-based file explorer
- File creation and modification
- Workspace organization

## Technical Architecture

### State Management
The application uses React's useState for managing various states:
- File system state (files and directory structure)
- Open files and active file tracking
- Chat messages and terminal outputs
- UI state (active tabs and panels)

### Socket Communication
```
typescript
const socket = io('http://localhost:8000', {
transports: ['websocket'],
reconnection: true
});
```

Handles real-time communication with the backend server for:
- Chat messages
- File updates
- Terminal commands
- Agent actions

### Key Components

1. **App.tsx** (Main Container)
```
typescript:nodes/frontend/src/App.tsx
startLine: 37
endLine: 322
```
Serves as the main application container, managing:
- Global state
- Component coordination
- Socket event handling
- File system operations

2. **ChatInterface**
```
typescript:nodes/frontend/src/components/ChatInterface/ChatInterface.tsx
startLine: 35
endLine: 129
```
Handles:
- Message display and parsing
- Real-time chat updates
- Visual notifications
- User input

3. **CodeEditor**
Provides:
- Multi-file editing
- Syntax highlighting
- Tab management
- File content updates

4. **FileSystem**
Features:
- Tree view of files
- File selection
- Directory navigation
- File type icons

## State Flow

1. **File Operations**
- File selection triggers state updates
- Content changes update both local and global state
- Real-time sync between components

2. **Chat Communication**
- Messages trigger UI updates
- Agent actions cause state changes
- Visual indicators for new messages

3. **Terminal Integration**
- Command outputs update terminal state
- Real-time display of execution results

## Getting Started

1. **Installation**
```bash
npm install
```


2. **Development**
```bash
npm start
```

Runs the app in development mode at http://localhost:3000

3. **Building**
```bash
npm run build
```

Creates production build in `build` folder

## Component Communication

The application uses a combination of:
- Props for component-to-component communication
- Socket events for real-time updates
- State management for UI synchronization

## Styling

- Custom CSS modules for component styling
- Tailwind CSS for utility classes
- Dark theme optimized for code editing

## Future Improvements

1. **Performance Optimization**
- Implement virtual scrolling for large files
- Optimize socket connections
- Add file content caching

2. **Feature Additions**
- Collaborative editing
- More language support
- Enhanced terminal features

3. **UI Enhancements**
- Customizable themes
- Resizable panels
- Enhanced file preview
