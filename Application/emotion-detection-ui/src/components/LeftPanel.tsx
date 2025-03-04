import React, { useRef, useState } from 'react';

const LeftPanel: React.FC = () => {
  const [imageURL, setImageURL] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    setImageURL(url);
  };

  return (
    <div className="left-panel">
      <button onClick={() => fileInputRef.current?.click()}>
        Upload Image
      </button>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept="image/*"
        onChange={handleFileChange}
      />
      {imageURL && (
        <div
          className="image-preview"
          style={{ backgroundImage: `url(${imageURL})` }}
        ></div>
      )}
    </div>
  );
};

export default LeftPanel;
