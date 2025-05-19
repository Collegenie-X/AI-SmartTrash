import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite';

export class ModelService {
  private static instance: ModelService;
  private tfliteModel: any = null;
  private labels: string[] = [];
  private modelBuffer: ArrayBuffer | null = null;

  private constructor() {}

  static getInstance(): ModelService {
    if (!ModelService.instance) {
      ModelService.instance = new ModelService();
    }
    return ModelService.instance;
  }

  async loadModel(modelBuffer: ArrayBuffer): Promise<void> {
    try {
      this.modelBuffer = modelBuffer;
      // TFLite 모델 초기화
      await tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
      this.tfliteModel = await tflite.loadTFLiteModel(modelBuffer);
      console.log('TFLite model loaded successfully');
    } catch (error) {
      console.error('Error loading TFLite model:', error);
      throw new Error('Failed to initialize TFLite model');
    }
  }

  setLabels(labels: string[]): void {
    this.labels = labels;
    console.log('Labels set successfully:', labels);
  }

  async predict(input: tf.Tensor): Promise<{
    label: string;
    confidence: number;
  }> {
    if (!this.tfliteModel || this.labels.length === 0) {
      throw new Error('Model or labels not loaded');
    }

    try {
      // 모델 추론 실행
      const outputTensor = await this.tfliteModel.predict(input);
      const scores = await outputTensor.data();
      
      // 가장 높은 확률을 가진 클래스 찾기
      const maxScore = Math.max(...scores);
      const maxScoreIndex = scores.indexOf(maxScore);
      
      return {
        label: this.labels[maxScoreIndex],
        confidence: maxScore
      };
    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error('Failed to run prediction');
    }
  }

  isModelReady(): boolean {
    return this.tfliteModel !== null && this.labels.length > 0;
  }

  getLabels(): string[] {
    return this.labels;
  }
} 