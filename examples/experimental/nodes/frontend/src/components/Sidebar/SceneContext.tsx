import React from 'react';
import './SceneContext.css';

interface SceneContextProps {
  messages: { text: string; agentName?: string }[];
}

export const SceneContext: React.FC<SceneContextProps> = ({ messages }) => {
  console.log(messages);
  return (
    <div id="scene-context-container">
      <div id="scene-context-header">Scene Context</div>
      <div id="scene-context-messages">
        {messages.map((message, index) => (
          <div key={index} className="scene-message">
            {message.agentName && <strong>{message.agentName}: </strong>}
            {message.text}
          </div>
        ))}
      </div>
    </div>
  );
};
