import React from 'react';

interface ProgressBarProps {
  value: number;
  color: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ value, color }) => {
  return (
    <div
      style={{
        width: '150px',
        height: '12px',
        backgroundColor: '#e0e0e0',
        borderRadius: '6px',
        overflow: 'hidden',
        margin: '0 0.5rem'
      }}
    >
      <div style={{ width: `${value}%`, height: '100%', backgroundColor: color }} />
    </div>
  );
};

const EmotionResult: React.FC = () => {
  const EMOTION_LIST = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprised',
  ];

  // Dummy data for demonstration; later this data will come as props.
  const dummyEmotions: Record<string, number> = { happy: 60, neutral: 15, surprised: 5, angry: 5, disgust: 5, fear: 5, sad: 5 };

  // Define subdued colors for each emotion.
  const emotionColors: Record<string, string> = {
    angry: '#d9534f',     // red
    disgust: '#5cb85c',   // green
    fear: '#9370DB',      // medium purple
    happy: '#f0ad4e',     // orange
    neutral: '#9E9E9E',   // gray
    sad: '#5bc0de',       // light blue
    surprised: '#f7e359', // soft yellow
  };

  return (
    <div className="emotion-result">
      <h3>Emotion Results</h3>
      {EMOTION_LIST.map((emo) => {
        const value = dummyEmotions[emo] || 0;
        return (
          <div
            key={emo}
            className="emotion-row"
            style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center' }}
          >
            <label
              className="emotion-label"
              style={{ width: '80px', display: 'inline-block', color: '#eee' }}
            >
              {capitalizeFirstChar(emo)}
            </label>
            <ProgressBar value={value} color={emotionColors[emo]} />
            <span style={{ marginLeft: '0.5rem', color: '#bbb', fontSize: '0.85rem' }}>
              {value}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

function capitalizeFirstChar(str: string) {
  if (!str) return str; // Check if the string is empty
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export default EmotionResult;
