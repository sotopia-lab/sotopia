import React from 'react';
import { FaCogs, FaFolderOpen } from 'react-icons/fa'; // Import the file icon
import './Sidebar.css';

interface SidebarProps {
    onSelect: (option: 'fileSystem' | 'sceneContext') => void;
  }

export const Sidebar: React.FC<SidebarProps> = ({ onSelect }) => {
  return (
    <div className="sidebar">
      <button className="sidebar-button" onClick={() => onSelect('fileSystem')}>
        <FaFolderOpen size={24} />
      </button>
      <button className="sidebar-button" onClick={() => onSelect('sceneContext')}>
        <FaCogs size={24} />
      </button>
      {/* Add more icons as needed */}
    </div>
  );
};