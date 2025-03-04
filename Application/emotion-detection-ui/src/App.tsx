import React from 'react';
import './App.css';
import CheckBoxPanel from './components/CheckBoxPanel';
import EmotionResult from './components/EmotionResult';
import LeftPanel from './components/LeftPanel';

const App: React.FC = () => {
  return (
    <div className="app-container">
      {/* Left Panel (2/3 width) */}
      <div className="left-panel">
        <LeftPanel />
      </div>

      {/* Right Panel (1/3 width) */}
      <div className="right-panel">
        <CheckBoxPanel />
        <EmotionResult />
      </div>
    </div>
  );
};

export default App;
