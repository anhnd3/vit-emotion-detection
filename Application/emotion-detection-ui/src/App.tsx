import React, { useState } from 'react';
import './App.css';
import CheckBoxPanel from './components/CheckBoxPanel';
import EmotionResult from './components/EmotionResult';
import LeftPanel from './components/LeftPanel';

const App: React.FC = () => {
  const [emotionResults, setEmotionResults] = useState<{ [key: string]: number }>({});
  const [selectedModel, setSelectedModel] = useState<string>('vit_default');

  const handleResult = (results: { [key: string]: number }) => {
    setEmotionResults(results);
  };

  return (
    <div className="app-container">
      {/* Left Panel (2/3 width) */}
      <div className="left-panel">
        <LeftPanel onResult={handleResult} selectedModel={selectedModel} />
      </div>

      {/* Right Panel (1/3 width) */}
      <div className="right-panel">
        <div className="project-title">
          <h2>Emotion Detection Demo</h2>
        </div>
        <CheckBoxPanel />
        <div className="model-selector" style={{ marginBottom: '1rem' }}>
          <h4>Select Model</h4>
          <label>
            <input
              type="radio"
              value="vit_default"
              checked={selectedModel === 'vit_default'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            ViT Default
          </label>
          <br />
          <label>
            <input
              type="radio"
              value="resnet50"
              checked={selectedModel === 'resnet50'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            Resnet
          </label>
          <br />
          <label>
            <input
              type="radio"
              value="vit_2"
              checked={selectedModel === 'vit_optimized'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            ViT 2
          </label>
        </div>
        <EmotionResult emotionResults={emotionResults} />
      </div>
    </div>
  );
};

export default App;
