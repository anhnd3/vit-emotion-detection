import React, { useState } from 'react';
import './App.css';
import CheckBoxPanel from './components/CheckBoxPanel';
import EmotionResult from './components/EmotionResult';
import LeftPanel from './components/LeftPanel';

const App: React.FC = () => {
  // Global state to store annotated image (as base64) and emotion state map.
  const [annotatedImage, setAnnotatedImage] = useState<string>('');
  const [emotionResults, setEmotionResults] = useState<{ [key: string]: number }>({});

  // Callback to update the global state when new results are received.
  const handleResult = (img: string, results: { [key: string]: number }) => {
    setAnnotatedImage(img);
    setEmotionResults(results);
  };

  return (
    <div className="app-container">
      {/* Left Panel (2/3 width) */}
      <div className="left-panel">
        <LeftPanel onResult={handleResult} annotatedImage={annotatedImage} />
      </div>

      {/* Right Panel (1/3 width) */}
      <div className="right-panel">
      <div className="project-title">
          <h2>ViT - Emotion Detection</h2>
        </div>
        <CheckBoxPanel />
        <EmotionResult emotionResults={emotionResults} />
      </div>
    </div>
  );
};

export default App;
