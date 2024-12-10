/**
 * Browser.tsx
 *
 * This component represents a simple web browser interface within the application. It allows
 * users to enter a URL and navigate to that page. The browser displays the content of the
 * specified URL in an iframe, enabling users to interact with web content directly within
 * the application.
 *
 * Key Features:
 * - Provides an input field for entering URLs.
 * - Includes a "Go" button to navigate to the entered URL.
 * - Displays the web content in an iframe.
 *
 * Props:
 * - url: The initial URL to be loaded in the browser.
 *
 */
"use client"

import React, { useState } from 'react';
import '../../styles/globals.css';

// Define the props for the Browser component
interface BrowserProps {
  url: string; // The initial URL to load
}

// Main Browser component definition
export const Browser: React.FC<BrowserProps> = ({ url }) => {
    const [currentUrl, setCurrentUrl] = useState(url); // State to manage the current URL

    // const proxyUrl = `https://cors-anywhere.herokuapp.com/${currentUrl}`;

    return (
      <div className="browser-container">
        <div className="browser-toolbar">
          <input
            value={currentUrl} // Bind input value to currentUrl state
            onChange={(e) => setCurrentUrl(e.target.value)} // Update currentUrl on input change
            placeholder="Enter URL" // Placeholder text for the input field
          />
          <button onClick={() => setCurrentUrl(currentUrl)}>Go</button> {/* Button to navigate to the current URL */}
        </div>
        <div className="browser-content">
          <iframe src={currentUrl} title="Browser Window" /> {/* Display the web content in an iframe */}
        </div>
      </div>
    );
};
