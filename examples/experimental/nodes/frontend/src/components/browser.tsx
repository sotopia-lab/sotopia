// examples/experimental/nodes/frontend/src/components/browser.tsx
import React, { useState } from 'react';


interface BrowserProps {
  url: string;
}

export const Browser: React.FC<BrowserProps> = ({ url }) => {
    const [currentUrl, setCurrentUrl] = useState(url);
  
    return (
      <div className="browser-container">
        <div className="browser-toolbar">
          <input
            value={url}
            onChange={(e) => setCurrentUrl(e.target.value)}
            placeholder="Enter URL"
          />
          <button onClick={() => {}}>Go</button>
        </div>
        <div className="browser-content">
          <iframe src={url} title="Browser Window" />
        </div>
      </div>
    );
  };
