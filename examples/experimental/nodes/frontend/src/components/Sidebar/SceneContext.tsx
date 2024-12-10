/**
 * SceneContext.tsx
 *
 * This component displays the context messages from agents within the application. It allows
 * users to view messages that provide additional information or context related to the current
 * scene or interaction. The messages are rendered in a markdown format for better readability.
 *
 * Key Features:
 * - Displays a header for the scene context.
 * - Renders messages with optional agent names.
 * - Supports markdown formatting for message content.
 *
 * Props:
 * - messages: An array of message objects, each containing text and an optional agent name.
 *
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import '../../styles/globals.css';

// Define the structure of the props for the SceneContext component
interface SceneContextProps {
  messages: { text: string; agentName?: string }[];
}

// Main SceneContext component definition
export const SceneContext: React.FC<SceneContextProps> = ({ messages }) => {
  console.log(messages); // Log messages for debugging purposes

  return (
    <div id="scene-context-container">
      <div id="scene-context-header">Scene Context</div>
      <div id="scene-context-messages">
        {messages.map((message, index) => (
          <div key={index} className="scene-message">
            {message.agentName && <strong>{message.agentName}: </strong>} {/* Display agent name if available */}
            <ReactMarkdown>{message.text}</ReactMarkdown> {/* Render message text as markdown */}
          </div>
        ))}
      </div>
    </div>
  );
};
