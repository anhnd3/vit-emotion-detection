import React from 'react';

const CheckBoxPanel: React.FC = () => {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <h3>Face Analysis</h3>
      <label>
        <input type="checkbox" disabled />
        Gender
      </label>
      <br />
      <label>
        <input type="checkbox" disabled />
        Age
      </label>
      <br />
      <label>
        <input type="checkbox" checked readOnly />
        Emotions
      </label>
    </div>
  );
};

export default CheckBoxPanel;
