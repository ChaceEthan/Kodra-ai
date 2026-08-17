import React from "react";
import { Cpu, Layers, Zap, Database, GitBranch, ArrowRight, ShieldCheck, Box } from "lucide-react";
import { ModelConfigUI } from "../types";

interface ArchitectureViewProps {
  config: ModelConfigUI;
  setConfig: React.Dispatch<React.SetStateAction<ModelConfigUI>>;
}

export const ArchitectureView: React.FC<ArchitectureViewProps> = ({ config, setConfig }) => {
  // Real dynamic parameter calculations based on exact PyTorch Causal GPT formula
  const tokenEmbedParams = config.vocabSize * config.embeddingDim;
  const posEmbedParams = config.contextLength * config.embeddingDim;
  
  // Multi-head Causal Attention Params (c_attn projection: 3 * d_model * d_model + 3 * d_model, c_proj: d_model * d_model + d_model)
  const cAttnWeight = config.embeddingDim * (3 * config.embeddingDim);
  const cAttnBias = 3 * config.embeddingDim;
  const cProjWeight = config.embeddingDim * config.embeddingDim;
  const cProjBias = config.embeddingDim;
  const attnParamsPerLayer = cAttnWeight + cAttnBias + cProjWeight + cProjBias;

  // MLP FeedForward Params (c_fc: d_model * 4d_model + 4d_model, c_proj: 4d_model * d_model + d_model)
  const cFcWeight = config.embeddingDim * (4 * config.embeddingDim);
  const cFcBias = 4 * config.embeddingDim;
  const cMlpProjWeight = (4 * config.embeddingDim) * config.embeddingDim;
  const cMlpProjBias = config.embeddingDim;
  const mlpParamsPerLayer = cFcWeight + cFcBias + cMlpProjWeight + cMlpProjBias;

  // LayerNorm Params (ln_1: 2 * d_model, ln_2: 2 * d_model)
  const lnParamsPerLayer = 2 * (2 * config.embeddingDim);

  const blockParamsPerLayer = attnParamsPerLayer + mlpParamsPerLayer + lnParamsPerLayer;
  const totalBlockParams = config.transformerLayers * blockParamsPerLayer;

  // Final LayerNorm (ln_f: 2 * d_model)
  const finalLnParams = 2 * config.embeddingDim;

  // Total trainable parameters (Tied embeddings share wte with lm_head)
  const totalParams = tokenEmbedParams + posEmbedParams + totalBlockParams + finalLnParams;

  return (
    <div className="space-[#1e293b] space-y-6">
      {/* Primary Banner Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-2xl shrink-0 shadow-lg">
              <img
                src="/public/branding/kodra_logo.svg"
                alt="Kodra AI"
                className="h-12 w-auto object-contain"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
            <div>
              <div className="flex items-center space-x-3">
                <h2 className="text-2xl font-black tracking-tight text-white">KODRA AI</h2>
                <span className="bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                  Phase 1 PyTorch Core
                </span>
              </div>
              <p className="text-slate-300 text-sm mt-1 font-medium">
                Scratch PyTorch Causal GPT Code Model • Your Intelligent Coding Partner
              </p>
              <p className="text-slate-400 text-xs mt-2 max-w-2xl leading-relaxed">
                Kodra GPT is a transparent, research-oriented Causal Transformer code generation model implemented completely from scratch in PyTorch. Features pre-LayerNorm architecture, GELU activation, tied token embeddings, and multi-head causal attention.
              </p>
            </div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl text-right flex flex-col justify-center min-w-[200px]">
            <span className="text-xs uppercase font-mono tracking-widest text-slate-400">Total Trainable Parameters</span>
            <span className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400 font-mono my-0.5">
              {(totalParams / 1e6).toFixed(2)}M
            </span>
            <span className="text-[11px] text-slate-500 font-mono">{totalParams.toLocaleString()} Parameters</span>
          </div>
        </div>
      </div>

      {/* Model Spec Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Context Length</span>
          <span className="text-lg font-bold font-mono text-cyan-400">{config.contextLength} tokens</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Embedding Dim</span>
          <span className="text-lg font-bold font-mono text-blue-400">{config.embeddingDim}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Attention Heads</span>
          <span className="text-lg font-bold font-mono text-emerald-400">{config.attentionHeads}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Transformer Layers</span>
          <span className="text-lg font-bold font-mono text-indigo-400">{config.transformerLayers}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Head Dimension</span>
          <span className="text-lg font-bold font-mono text-purple-400">{config.embeddingDim / config.attentionHeads}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <span className="text-xs text-slate-400 block font-medium">Vocabulary Size</span>
          <span className="text-lg font-bold font-mono text-amber-400">{config.vocabSize} chars</span>
        </div>
      </div>

      {/* Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Embeddings */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 mb-3">
              <Database className="w-5 h-5" />
              <h3 className="font-bold text-white">Embeddings & Projections</h3>
            </div>
            <ul className="space-y-2 text-xs font-mono text-slate-300">
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>Token Embedding (WTE):</span>
                <span className="text-cyan-400">{tokenEmbedParams.toLocaleString()}</span>
              </li>
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>Positional Embedding (WPE):</span>
                <span className="text-cyan-400">{posEmbedParams.toLocaleString()}</span>
              </li>
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>LM Head (Tied Weights):</span>
                <span className="text-emerald-400">Tied with WTE</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400">
            Tied embeddings drastically reduce memory footprint while stabilizing token representations.
          </div>
        </div>

        {/* Card 2: Transformer Layer */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-blue-400 mb-3">
              <Layers className="w-5 h-5" />
              <h3 className="font-bold text-white">Transformer Layer (x{config.transformerLayers})</h3>
            </div>
            <ul className="space-y-2 text-xs font-mono text-slate-300">
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>Causal Self-Attention:</span>
                <span className="text-blue-400">{attnParamsPerLayer.toLocaleString()}</span>
              </li>
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>FeedForward MLP (GELU):</span>
                <span className="text-blue-400">{mlpParamsPerLayer.toLocaleString()}</span>
              </li>
              <li className="flex justify-between border-b border-slate-800/80 pb-1.5">
                <span>Pre-LayerNorms (LN1, LN2):</span>
                <span className="text-blue-400">{lnParamsPerLayer.toLocaleString()}</span>
              </li>
            </ul>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400">
            Per-layer parameter count: <span className="font-mono text-blue-300">{blockParamsPerLayer.toLocaleString()}</span>
          </div>
        </div>

        {/* Card 3: Identity & Architecture Notes */}
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-purple-400 mb-3">
              <ShieldCheck className="w-5 h-5" />
              <h3 className="font-bold text-white">Kodra Identity Matrix</h3>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              <p><span className="text-slate-400 font-semibold">Product:</span> Kodra AI</p>
              <p><span className="text-slate-400 font-semibold">Model:</span> Kodra GPT (KodraGPT)</p>
              <p><span className="text-slate-400 font-semibold">Core Engine:</span> Kodra Core</p>
              <p><span className="text-slate-400 font-semibold">Future Agent:</span> Kodra Agent (Phase 2/3)</p>
              <p><span className="text-slate-400 font-semibold">Future IDE:</span> Kodra for VS Code (Phase 3)</p>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400">
            Phase 1 is educational/research scale and fully modular for learning PyTorch Transformer internals.
          </div>
        </div>
      </div>
    </div>
  );
};
