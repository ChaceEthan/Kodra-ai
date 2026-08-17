export interface ModelConfigUI {
  contextLength: number;
  embeddingDim: number;
  attentionHeads: number;
  transformerLayers: number;
  dropout: number;
  vocabSize: number;
  batchSize: number;
  learningRate: number;
}

export interface SystemStatusIndicators {
  backend_connected: boolean;
  model_initialized: boolean;
  trained_checkpoint_loaded: boolean;
  dataset_loaded: boolean;
  training_running: boolean;
  checkpoint_available: boolean;
  inference_available: boolean;
}

export interface TokenizeResponse {
  success: boolean;
  text: string;
  tokens: number[];
  token_chars: string[];
  vocab_size: number;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k?: number;
}

export interface GenerateResponse {
  success: boolean;
  prompt: string;
  generated_text: string;
  full_output: string;
  tokens_generated: number;
  time_taken_sec: number;
}

export interface TrainingMetric {
  epoch: number;
  step: number;
  loss: number;
  perplexity: number;
  learning_rate: number;
  timestamp: string;
}

export interface TestResult {
  name: string;
  status: "PASSED" | "FAILED" | "SKIPPED";
  duration_ms: number;
  details?: string;
  error?: string;
}

export interface TestSuiteResponse {
  success: boolean;
  passed: number;
  failed: number;
  skipped: number;
  total: number;
  results: TestResult[];
}
