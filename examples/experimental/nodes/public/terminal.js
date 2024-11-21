// terminal.js
document.addEventListener('DOMContentLoaded', () => {
    const terminal = new Terminal({
        cursorBlink: true,
        theme: {
            background: '#000000',
            foreground: '#ffffff'
        }
    });
    terminal.open(document.getElementById('terminal'));

    // Buffer to store user input
    let inputBuffer = '';

    // Handle user input
    terminal.onData(data => {
        switch (data) {
            case '\u0003': // Ctrl+C
                terminal.write('^C');
                break;
            case '\r': // Enter
                terminal.write('\r\n');
                inputBuffer = ''; // Clear buffer on enter
                break;
            case '\u007F': // Backspace
                if (inputBuffer.length > 0) {
                    inputBuffer = inputBuffer.slice(0, -1);
                    terminal.write('\b \b');
                }
                break;
            default:
                inputBuffer += data;
                terminal.write(data);
        }
    });

    // Make terminal accessible globally
    window.myTerminal = terminal;
});