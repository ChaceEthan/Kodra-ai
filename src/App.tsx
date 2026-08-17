import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { ArchitectureView } from "./components/ArchitectureView";
import { FileExplorer } from "./components/FileExplorer";
import { TokenizerWorkbench } from "./components/TokenizerWorkbench";
import { AttentionVisualizer } from "./components/AttentionVisualizer";
import { TrainingDashboard } from "./components/TrainingDashboard";
import { GenerationPlayground } from "./components/GenerationPlayground";
import { TestRunner } from "./components/TestRunner";
import { ModelConfigUI, SystemStatusIndicators } from "./types";

export default function App() {
  const [activeTab, setActiveTab] = useState<string>("overview");

  // Global default configuration state for Kodra AI
  const [modelConfig, setModelConfig] = useState<ModelConfigUI>({
    contextLength: 256,
    embeddingDim: 256,
    attentionHeads: 4,
    transformerLayers: 6,
    dropout: 0.1,
    vocabSize: 96,
    batchSize: 16,
    learningRate: 0.0003,
  });

  const [backendStatus, setBackendStatus] = useState<any>(null);

  useEffect(() => {
    const fetchStatus = () => {
      fetch("/api/status")
        .then((res) => res.json())
        .then((data) => {
          setBackendStatus(data);
          if (data.config) {
            setModelConfig((prev) => ({
              ...prev,
              contextLength: data.config.context_length ?? prev.contextLength,
              embeddingDim: data.config.embedding_dim ?? prev.embeddingDim,
              attentionHeads: data.config.attention_heads ?? prev.attentionHeads,
              transformerLayers: data.config.transformer_layers ?? prev.transformerLayers,
              vocabSize: data.config.vocab_size ?? prev.vocabSize,
            }));
          }
        })
        .catch((err) => console.error("Error fetching status:", err));
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Calculate real parameter count dynamically based on KodraGPT PyTorch architecture
  const tokenEmbedParams = modelConfig.vocabSize * modelConfig.embeddingDim;
  const posEmbedParams = modelConfig.contextLength * modelConfig.embeddingDim;
  const attnParams = (modelConfig.embeddingDim * 3 * modelConfig.embeddingDim + 3 * modelConfig.embeddingDim) +
                     (modelConfig.embeddingDim * modelConfig.embeddingDim + modelConfig.embeddingDim);
  const mlpParams = (modelConfig.embeddingDim * (4 * modelConfig.embeddingDim) + 4 * modelConfig.embeddingDim) +
                    ((4 * modelConfig.embeddingDim) * modelConfig.embeddingDim + modelConfig.embeddingDim);
  const lnParams = 2 * (2 * modelConfig.embeddingDim);
  const blockParams = attnParams + mlpParams + lnParams;
  const totalBlockParams = modelConfig.transformerLayers * blockParams;
  const finalLnParams = 2 * modelConfig.embeddingDim;
  const totalParams = backendStatus?.param_count ?? (tokenEmbedParams + posEmbedParams + totalBlockParams + finalLnParams);

  const formattedParamCount = `${(totalParams / 1e6).toFixed(2)}M`;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* App Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        paramCount={formattedParamCount}
        indicators={backendStatus?.indicators}
        device={backendStatus?.device}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "overview" && (
          <ArchitectureView config={modelConfig} setConfig={setModelConfig} />
        )}
        {activeTab === "explorer" && <FileExplorer />}
        {activeTab === "tokenizer" && <TokenizerWorkbench />}
        {activeTab === "attention" && <AttentionVisualizer />}
        {activeTab === "training" && <TrainingDashboard />}
        {activeTab === "generation" && <GenerationPlayground />}
        {activeTab === "tests" && <TestRunner />}
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
        Kodra AI • Production Causal GPT Model Architecture Built From Scratch
      </footer>
    </div>
  );
}
