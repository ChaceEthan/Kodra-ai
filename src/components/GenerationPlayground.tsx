import React, { useState } from "react";
import { Play, Sparkles, Copy, Check } from "lucide-react";

export const GenerationPlayground: React.FC = () => {
  const [prompt, setPrompt] = useState<string>("def fibonacci(n: int) -> int:");
  const [maxTokens, setMaxTokens] = useState<number>(64);
  const [temp, setTemp] = useState<number>(0.7);
  const [generatedText, setGeneratedText] = useState<string>("");
  const [checkpointNote, setCheckpointNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const handleGenerate = () => {
    setLoading(true);
    setError(null);
    fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_new_tokens: maxTokens,
        temperature: temp,
        top_k: 40,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.full_output) {
          setGeneratedText(data.full_output);
          setCheckpointNote(data.note || null);
        } else {
          setGeneratedText("");
          setError(data.error || "Generation failed - no output returned by the backend.");
        }
      })
      .catch(() => {
        setGeneratedText("");
        setError("Could not reach the Kodra AI Agent backend. Is the FastAPI server running?");
      })
      .finally(() => setLoading(false));
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cyan-400" /> Kodra GPT Code Completion Playground
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Autoregressive Causal Inference Engine for Kodra GPT
          </p>
        </div>
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="flex items-center space-x-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-5 py-2 rounded-xl text-xs font-semibold shadow-md transition-all"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>{loading ? "Generating Code..." : "Complete Code"}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="space-y-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
          <div>
            <label className="text-xs font-mono text-slate-400 block mb-1">Temperature: {temp}</label>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.1"
              value={temp}
              onChange={(e) => setTemp(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-slate-400 block mb-1">Max Tokens: {maxTokens}</label>
            <input
              type="range"
              min="16"
              max="256"
              step="16"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-cyan-400"
            />
          </div>
        </div>

        {/* Prompt Input */}
        <div className="space-y-2 md:col-span-2">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
            Prompt Prefix
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={5}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50 resize-none"
          />
        </div>
      </div>

      {/* Output Panel */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-mono uppercase tracking-wider text-slate-400">
            Model Generation Output
          </label>
          {generatedText && (
            <button
              onClick={copyToClipboard}
              className="flex items-center space-x-1 text-xs text-slate-400 hover:text-white transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
          )}
        </div>
        {checkpointNote && (
          <div className="text-[11px] font-mono text-amber-400 bg-amber-950/30 border border-amber-800/50 rounded-lg px-3 py-1.5">
            {checkpointNote}
          </div>
        )}
        {error && (
          <div className="text-[11px] font-mono text-red-400 bg-red-950/30 border border-red-800/50 rounded-lg px-3 py-1.5">
            {error}
          </div>
        )}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 min-h-[160px] font-mono text-xs text-cyan-300 whitespace-pre">
          {generatedText || <span className="text-slate-600">Generated code output will appear here after inference...</span>}
        </div>
      </div>
    </div>
  );
};
