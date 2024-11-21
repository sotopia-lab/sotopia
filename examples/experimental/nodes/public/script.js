// Initialize files object to store file paths and content
const files = {
    "/workspace/index.html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <title>Document</title>\n</head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>",
    "/workspace/style.css": "body {\n    background-color: #f0f0f0;\n    font-family: Arial, sans-serif;\n}",
    "/workspace/script.js": "console.log('Hello, World!');",
    "/workspace/interview.py": "# Python code here"
};

// Populate file list
const fileList = document.getElementById('file-list');

// Create the "workspace" folder and append it to the file list
const workspaceFolder = document.createElement('li');
workspaceFolder.textContent = 'workspace';
const workspaceFileList = document.createElement('ul');
workspaceFolder.appendChild(workspaceFileList);
fileList.appendChild(workspaceFolder);

Object.keys(files).forEach(filePath => {
    addFileToExplorer(filePath);
});

// Function to add a file to the file explorer
function addFileToExplorer(filePath) {
    const parts = filePath.split('/').filter(part => part);

    if (parts[0] === "workspace") {
        // If the file is within the workspace, add it to the workspaceFileList
        const fileName = parts[1];
        const li = document.createElement('li');
        li.textContent = fileName;
        li.addEventListener('click', () => loadFileContent(filePath));
        workspaceFileList.appendChild(li);
    }
}

// Load file content into the editor
function loadFileContent(fileName) {
    const editor = document.getElementById('editor');
    editor.value = files[fileName];
    editor.readOnly = false;
}

// Function to update or add a file
function updateFile(filePath, fileContent) {
    if (!files[filePath]) {
        addFileToExplorer(filePath);
    }
    files[filePath] = fileContent;
    loadFileContent(filePath);
}