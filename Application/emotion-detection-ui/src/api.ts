export async function sendImageToServer(file: File): Promise<Record<string, number>> {
    const formData = new FormData();
    formData.append('file', file);
  
    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch emotion data');
    }
    return response.json();
  }
  