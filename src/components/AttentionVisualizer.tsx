import React, { useState } from "react";
import { Sparkles, RefreshCw } from "lucide-react";

export const AttentionVisualizer: React.FC = () => {
  const sampleTokens = ["def", " ", "kodra", "_", "gpt", "(", "x", ")", ":"];
  const [selectedHead, setSelectedHead] = useState<number>(0);

  // Generate synthetic causal attention matrix (lower triangular)
  const generateCausalMatrix = () => {
    return sampleTokens.map((_, i) =>
      sampleTokens.map((_, j) => {
        if (j > i) return 0; // Causal mask enforces zero attention to future tokens
        const raw = Math.exp(-Math.abs(i - j) * 0.4 + (selectedHead * 0.1));
        return parseFloat(raw.toFixed(3));
      })
    );
  };

  const matrix = generateCausalMatrix();

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cyan-400" /> Causal Multi-Head Self-Attention
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Visualizing lower-triangular causal mask and attention weights across heads
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {[0, 1, 2, 3].map((head) => (
            <button
              key={head}
              onClick={() => setSelectedHead(head)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                selectedHead === head
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-bold"
                  : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-white"
              }`}
            >
              Head #{head + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="bg-slate-950 border border-slate-800 p-6 rounded-2xl overflow-x-auto">
        <div className="min-w-[500px]">
          <div className="grid grid-cols-10 gap-1 mb-2 font-mono text-[11px] text-slate-400 text-center">
            <div></div>
            {sampleTokens.map((tok, idx) => (
              <div key={idx} className="font-semibold text-cyan-400 truncate px-1">{tok}</div>
            ))}
          </div>

          {sampleTokens.map((rowTok, rowIdx) => (
            <div key={rowIdx} className="grid grid-cols-10 gap-1 my-1 items-center font-mono text-[11px]">
              <div className="font-semibold text-cyan-400 text-right pr-2 truncate">{rowTok}</div>
              {matrix[rowIdx].map((val, colIdx) => {
                const opacity = val;
                const isCausalMasked = colIdx > rowIdx;
                return (
                  <div
                    key={colIdx}
                    style={{
                      backgroundColor: isCausalMasked ? "#0f172a" : `rgba(6, 182, 212, ${Math.max(0.1, opacity)})`,
                    }}
                    className={`h-9 rounded flex items-center justify-center text-[10px] font-mono border ${
                      isCausalMasked ? "border-slate-900 text-slate-700" : "border-cyan-500/20 text-white font-semibold"
                    }`}
                  >
                    {isCausalMasked ? "-" : val.toFixed(2)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl text-xs text-slate-400 leading-relaxed font-mono">
        <span className="text-cyan-400 font-bold">Causal Guarantee:</span> Upper triangular elements are strictly masked to <span className="text-red-400 font-bold">-inf</span> before Softmax, guaranteeing that token <span className="text-white">t</span> can only attend to past positions <span className="text-white font-bold">&le; t</span> during autoregressive code generation.
      </div>
    </div>
  );
};
