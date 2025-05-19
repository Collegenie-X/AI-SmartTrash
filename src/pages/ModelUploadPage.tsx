import React, { useState } from 'react';
import ModelFileUploader from '../components/ModelUploader/ModelFileUploader';
import { ModelFiles, readLabelsFile, loadTFLiteModel, validateModelFiles } from '../utils/modelFileHandler';
import { ModelService } from '../services/ModelService';

const ModelUploadPage: React.FC = () => {
  const [modelFiles, setModelFiles] = useState<ModelFiles>({
    modelFile: null,
    labelsFile: null,
  });
  const [isModelReady, setIsModelReady] = useState(false);

  const handleModelUpload = async (file: File) => {
    try {
      setModelFiles(prev => ({ ...prev, modelFile: file }));
      
      // 모델 파일을 ArrayBuffer로 로드
      const modelBuffer = await loadTFLiteModel(file);
      
      // ModelService에 모델 로드
      const modelService = ModelService.getInstance();
      await modelService.loadModel(modelBuffer);
      
      console.log('Model loaded successfully');
      
      // 라벨도 이미 로드되어 있다면 모델 준비 상태 업데이트
      if (modelService.getLabels().length > 0) {
        setIsModelReady(true);
      }
    } catch (error) {
      console.error('Error handling model upload:', error);
      alert('Failed to load model file');
    }
  };

  const handleLabelsUpload = async (file: File) => {
    try {
      setModelFiles(prev => ({ ...prev, labelsFile: file }));
      
      // 라벨 파일 읽기
      const labels = await readLabelsFile(file);
      
      // ModelService에 라벨 설정
      const modelService = ModelService.getInstance();
      modelService.setLabels(labels);
      
      console.log('Labels loaded successfully:', labels);
      
      // 모델도 이미 로드되어 있다면 모델 준비 상태 업데이트
      if (modelService.isModelReady()) {
        setIsModelReady(true);
      }
    } catch (error) {
      console.error('Error handling labels upload:', error);
      alert('Failed to load labels file');
    }
  };

  const handleProcessFiles = async () => {
    const { modelFile, labelsFile } = modelFiles;
    
    if (!modelFile || !labelsFile) {
      alert('Please upload both model and labels files');
      return;
    }

    try {
      // 파일 유효성 검사
      validateModelFiles(modelFile, labelsFile);
      
      const modelService = ModelService.getInstance();
      if (!modelService.isModelReady()) {
        throw new Error('Model is not ready for processing');
      }
      
      // 여기에서 모델 사용 예시를 보여줄 수 있습니다
      console.log('Model is ready for processing');
      console.log('Available labels:', modelService.getLabels());
      
    } catch (error) {
      console.error('Error processing files:', error);
      alert(error instanceof Error ? error.message : 'Failed to process files');
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Upload Model Files</h1>
      
      <ModelFileUploader
        onModelUpload={handleModelUpload}
        onLabelsUpload={handleLabelsUpload}
      />

      <div className="mt-6">
        <button
          onClick={handleProcessFiles}
          disabled={!isModelReady}
          className={`px-6 py-3 rounded-lg ${
            !isModelReady
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700'
          } text-white font-semibold`}
        >
          Process Files
        </button>
        
        {isModelReady && (
          <p className="mt-2 text-green-600">
            Model and labels are ready for use!
          </p>
        )}
      </div>
    </div>
  );
};

export default ModelUploadPage; 