import React, { useRef, useState } from 'react';

interface LeftPanelProps {
  onResult: (annotatedImage: string, emotionResults: { [key: string]: number }) => void;
  annotatedImage: string;
}

const LeftPanel: React.FC<LeftPanelProps> = ({ onResult, annotatedImage }) => {
  const [imageURL, setImageURL] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }
    const url = URL.createObjectURL(file);
    setImageURL(url);

    // Prevent spamming: disable upload while processing.
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Upload failed with status " + response.status);
      }
      const data = await response.json();
      onResult(data.annotated_image, data.emotion_results);
    } catch (error) {
      console.error("Error during file upload:", error);
      alert("Failed to upload image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="left-panel">
      <button onClick={() => fileInputRef.current?.click()} disabled={loading}>
        {loading ? "Processing..." : "Upload Image"}
      </button>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        accept="image/*"
        onChange={handleFileChange}
      />
      {annotatedImage ? (
        <div
          className="image-preview"
          style={{
            backgroundImage: `url(data:image/webp;base64,${annotatedImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            width: "100%",
            height: "100%",
          }}
        />
      ) : (
        imageURL && (
          <div
            className="image-preview"
            style={{
              backgroundImage: `url(${imageURL})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
              width: "100%",
              height: "100%",
            }}
          />
        )
      )}
    </div>
  );
};

export default LeftPanel;
