import React, { useRef, useState } from 'react';

interface ModelFileUploaderProps {
  onModelUpload: (modelFile: File) => void;
  onLabelsUpload: (labelsFile: File) => void;
}

const ModelFileUploader: React.FC<ModelFileUploaderProps> = ({
  onModelUpload,
  onLabelsUpload,
}) => {
  const [modelFileName, setModelFileName] = useState<string>('');
  const [labelsFileName, setLabelsFileName] = useState<string>('');
  
  const modelInputRef = useRef<HTMLInputElement>(null);
  const labelsInputRef = useRef<HTMLInputElement>(null);

  const handleModelFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    if (!file.name.endsWith('.tflite')) {
      alert('Please upload a valid .tflite file');
      return;
    }
    
    setModelFileName(file.name);
    onModelUpload(file);
  };

  const handleLabelsFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    if (!file.name.endsWith('.txt')) {
      alert('Please upload a valid .txt file');
      return;
    }
    
    setLabelsFileName(file.name);
    onLabelsUpload(file);
  };

  return (
    <div className="p-4 border rounded-lg bg-white shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Model Files Upload</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            TFLite Model File (.tflite)
          </label>
          <div className="flex items-center">
            <input
              type="file"
              ref={modelInputRef}
              accept=".tflite"
              onChange={handleModelFileChange}
              className="hidden"
            />
            <button
              onClick={() => modelInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Choose Model File
            </button>
            <span className="ml-3 text-sm text-gray-600">
              {modelFileName || 'No file selected'}
            </span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Labels File (.txt)
          </label>
          <div className="flex items-center">
            <input
              type="file"
              ref={labelsInputRef}
              accept=".txt"
              onChange={handleLabelsFileChange}
              className="hidden"
            />
            <button
              onClick={() => labelsInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Choose Labels File
            </button>
            <span className="ml-3 text-sm text-gray-600">
              {labelsFileName || 'No file selected'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelFileUploader; 