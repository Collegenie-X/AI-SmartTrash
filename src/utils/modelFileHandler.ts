export interface ModelFiles {
  modelFile: File | null;
  labelsFile: File | null;
}

export const readLabelsFile = async (file: File): Promise<string[]> => {
  try {
    const text = await file.text();
    return text.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);
  } catch (error) {
    console.error('Error reading labels file:', error);
    throw new Error('Failed to read labels file');
  }
};

export const loadTFLiteModel = async (file: File): Promise<ArrayBuffer> => {
  try {
    return await file.arrayBuffer();
  } catch (error) {
    console.error('Error loading TFLite model:', error);
    throw new Error('Failed to load TFLite model');
  }
};

export const validateModelFiles = (modelFile: File, labelsFile: File): boolean => {
  if (!modelFile.name.endsWith('.tflite')) {
    throw new Error('Invalid model file format. Please upload a .tflite file');
  }

  if (!labelsFile.name.endsWith('.txt')) {
    throw new Error('Invalid labels file format. Please upload a .txt file');
  }

  return true;
}; 