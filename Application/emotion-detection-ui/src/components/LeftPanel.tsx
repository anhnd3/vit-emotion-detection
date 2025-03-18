import React, { useRef, useState, useEffect } from 'react';

interface LeftPanelProps {
  onResult: (emotionResults: { [key: string]: number }) => void;
  selectedModel: string;
}

const LeftPanel: React.FC<LeftPanelProps> = ({ onResult, selectedModel }) => {
  const [imageURL, setImageURL] = useState<string>('');
  const [annotatedImage, setAnnotatedImage] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [videoActive, setVideoActive] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);

  // Refs for file input, video element, canvases, WebSocket and animation frame id.
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  // Use number for the annotation interval
  const annotationIntervalRef = useRef<number | null>(null);

  // Ref to control reconnection attempts
  const reconnectAttemptRef = useRef<number>(0);
  // Flag to ensure only one annotation request is in flight.
  const requestInFlight = useRef<boolean>(false);

  // IMAGE UPLOAD LOGIC
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }
    const url = URL.createObjectURL(file);
    setImageURL(url);

    // Stop video if active.
    if (videoActive) {
      setVideoActive(false);
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", selectedModel);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Upload failed with status " + response.status);
      }
      const data = await response.json();
      setAnnotatedImage(data.annotated_image);
      onResult(data.emotion_results);
    } catch (error) {
      console.error("Error during file upload:", error);
      alert("Failed to upload image.");
    } finally {
      setLoading(false);
    }
  };

  // Function to send one annotation request if none is in flight.
  const sendAnnotationRequest = () => {
    if (
      videoRef.current &&
      hiddenCanvasRef.current &&
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN &&
      !requestInFlight.current
    ) {
      const offscreenCtx = hiddenCanvasRef.current.getContext('2d');
      if (offscreenCtx) {
        offscreenCtx.drawImage(
          videoRef.current,
          0,
          0,
          hiddenCanvasRef.current.width,
          hiddenCanvasRef.current.height
        );
        const frameData = hiddenCanvasRef.current.toDataURL("image/jpeg", 0.7);
        wsRef.current.send(frameData);
        requestInFlight.current = true;
      }
    }
  };

  // Throttle annotation requests using setInterval.
  const startStreaming = () => {
    // Continuous preview update.
    // const updatePreview = () => {
    //   if (videoRef.current && displayCanvasRef.current) {
    //     const ctx = displayCanvasRef.current.getContext('2d');
    //     if (ctx) {
    //       ctx.drawImage(videoRef.current, 0, 0, displayCanvasRef.current.width, displayCanvasRef.current.height);
    //     }
    //   }
    //   requestAnimationFrame(updatePreview);
    // };
    // updatePreview();
    annotationIntervalRef.current = window.setInterval(() => {
      sendAnnotationRequest();
    }, 400);
  };

  const clearAnnotationInterval = () => {
    if (annotationIntervalRef.current) {
      clearInterval(annotationIntervalRef.current);
      annotationIntervalRef.current = null;
    }
  };

  // Disconnect WebSocket and cancel streaming.
  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearAnnotationInterval();
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
    }
  };

  // Define reconnectWebSocket and connectWebSocket as function declarations.
  function reconnectWebSocket() {
    if (!videoActive) return;
    if (reconnectAttemptRef.current > 5) {
      console.error("Max reconnect attempts reached, not retrying.");
      return;
    }
    reconnectAttemptRef.current += 1;
    console.log(`Reconnecting WebSocket... Attempt ${reconnectAttemptRef.current}`);
    setTimeout(() => {
      connectWebSocket();
    }, 1000);
  }

  function connectWebSocket() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return; // already connected
    wsRef.current = new WebSocket("ws://localhost:8000/ws");

    wsRef.current.onopen = () => {
      console.log("WebSocket connection opened");
      reconnectAttemptRef.current = 0;
      startStreaming();
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setFps(data.fps);
      requestInFlight.current = false;
      if (displayCanvasRef.current) {
        const ctx = displayCanvasRef.current.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            ctx.drawImage(img, 0, 0, displayCanvasRef.current!.width, displayCanvasRef.current!.height);
          };
          img.src = data.frame;
        }
      }
      setAnnotatedImage(data.frame);
      onResult(data.emotion_results);
    };

    wsRef.current.onerror = (err) => {
      console.error("WebSocket error:", err);
      reconnectWebSocket();
    };

    wsRef.current.onclose = (event) => {
      console.log("WebSocket connection closed", event);
      if (videoActive) {
        reconnectWebSocket();
      }
    };
  }

  useEffect(() => {
    if (videoActive) {
      reconnectAttemptRef.current = 0;
      (async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
          }
          connectWebSocket();
        } catch (err) {
          console.error("Camera access denied:", err);
          alert("Camera access denied");
        }
      })();
    } else {
      disconnectWebSocket();
    }
    return () => {
      disconnectWebSocket();
    };
  }, [videoActive]);

  return (
    <div className="left-panel">
      <div style={{ marginBottom: '1rem' }}>
        <button
          onClick={() => {
            setVideoActive(false);
            fileInputRef.current?.click();
          }}
          disabled={loading}
          style={{
            backgroundColor: !videoActive ? '#4CAF50' : '#ddd',
            color: !videoActive ? 'white' : 'black',
            padding: '10px 15px',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          {loading ? "Processing..." : "Upload Image"}
        </button>
        <button
          onClick={() => setVideoActive(true)}
          style={{
            marginLeft: '1rem',
            backgroundColor: videoActive ? '#4CAF50' : '#ddd',
            color: videoActive ? 'white' : 'black',
            padding: '10px 15px',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          Video Stream
        </button>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        accept="image/*"
        onChange={handleFileChange}
      />
      {videoActive ? (
        <div className="video-review" style={{ position: 'relative', width: '100%', height: '100%' }}>
          <video ref={videoRef} style={{ display: 'none' }} />
          <canvas
            ref={displayCanvasRef}
            width={640}
            height={480}
            style={{ width: '100%', height: '100%' }}
          />
          <canvas
            ref={hiddenCanvasRef}
            width={640}
            height={480}
            style={{ display: 'none' }}
          />
          <div
            style={{
              position: 'absolute',
              bottom: '10px',
              right: '10px',
              color: '#fff',
              background: 'rgba(0,0,0,0.5)',
              padding: '5px',
              fontFamily: 'sans-serif',
            }}
          >
            FPS: {fps.toFixed(2)}
          </div>
        </div>
      ) : (
        <>
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
        </>
      )}
    </div>
  );
};

export default React.memo(LeftPanel, (prevProps, nextProps) => {
  // Only re-render if selectedModel or onResult prop changes.
  return (
    prevProps.selectedModel === nextProps.selectedModel &&
    prevProps.onResult === nextProps.onResult
  );
});
