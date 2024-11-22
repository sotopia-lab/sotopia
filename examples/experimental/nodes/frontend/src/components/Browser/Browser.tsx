// examples/experimental/nodes/frontend/src/components/browser.tsx
import React, { useState } from 'react';
import './Browser.css'; // Import the CSS file

interface BrowserProps {
  url: string;
}

export const Browser: React.FC<BrowserProps> = ({ url }) => {
    const [currentUrl, setCurrentUrl] = useState(url);

    return (
      <div className="browser-container">
        <div className="browser-toolbar">
          <input
            value={currentUrl}
            onChange={(e) => setCurrentUrl(e.target.value)}
            placeholder="Enter URL"
          />
          <button onClick={() => setCurrentUrl(currentUrl)}>Go</button>
        </div>
        <div className="browser-content">
          <iframe src={currentUrl} title="Browser Window" />
        </div>
      </div>
    );
  };
