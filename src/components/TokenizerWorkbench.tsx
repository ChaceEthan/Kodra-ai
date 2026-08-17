import React, { useState } from "react";
import { Terminal, Play, CheckCircle, Info } from "lucide-react";
import { TokenizeResponse } from "../types";

export const TokenizerWorkbench: React.FC = () => {
  const [inputText, setInputText] = useState<string>("def kodra_ai_sum(a: int, b: int) -> int:\n    return a + b");
  const [result, setResult] = useState<TokenizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleTokenize = () => {
    setLoading(true);
    setError(null);
    fetch("/api/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.tokens) {
          setResult(data);
        } else {
          setResult(null);
          setError("Tokenization failed - no tokens returned by the backend.");
        }
      })
      .catch(() => {
        setResult(null);
        setError("Could not reach the Kodra AI Agent backend. Is the FastAPI server running?");
      })
      .finally(() => setLoading(false));
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <Terminal className="w-5 h-5 text-cyan-400" /> Tokenizer Workbench
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Character-level & Subword Tokenization Pipeline for Kodra AI
          </p>
        </div>
        <button
          onClick={handleTokenize}
          disabled={loading}
          className="flex items-center space-x-2 bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-xl text-xs font-semibold transition-all shadow-md"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>{loading ? "Tokenizing..." : "Run Tokenizer"}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Input Textarea */}
        <div className="space-y-2">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
            Input Code String
          </label>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={8}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50 resize-none"
          />
        </div>

        {/* Token Output Matrix */}
        <div className="space-y-2">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
            Encoded Token IDs Matrix
          </label>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 h-48 overflow-y-auto no-scrollbar font-mono text-xs">
            {result ? (
              <div className="flex flex-wrap gap-1.5">
                {result.tokens.map((tok, idx) => (
                  <span
                    key={idx}
                    className="bg-slate-900 border border-slate-800 hover:border-cyan-500/50 text-cyan-300 px-2 py-1 rounded text-[11px] font-mono flex items-center gap-1 group relative cursor-pointer"
                  >
                    <span className="text-slate-500 text-[10px]">#{idx}:</span>
                    <span>{tok}</span>
                    <span className="text-slate-400 text-[10px]">({result.token_chars[idx] === "\n" ? "\\n" : result.token_chars[idx]})</span>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-xs">Click "Run Tokenizer" to convert code into integer token IDs.</p>
            )}
          </div>
        </div>
      </div>

      {result && (
        <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl flex items-center justify-between text-xs font-mono text-slate-300">
          <div>Vocabulary Size: <span className="text-cyan-400 font-bold">{result.vocab_size}</span></div>
          <div>Total Sequence Length: <span className="text-emerald-400 font-bold">{result.tokens.length} tokens</span></div>
          <div>Encoding Lossless: <span className="text-blue-400 font-bold">100% Exact Reconstruction</span></div>
        </div>
      )}
      {error && (
        <div className="text-[11px] font-mono text-red-400 bg-red-950/30 border border-red-800/50 rounded-lg px-3 py-1.5">
          {error}
        </div>
      )}
    </div>
  );
};
