import React, { useState, useEffect, useRef } from "react";
import { Code2, Play, Square, RotateCcw, TrendingDown, Cpu } from "lucide-react";
import { TrainingMetric } from "../types";

export const TrainingDashboard: React.FC = () => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [checkpointAvailable, setCheckpointAvailable] = useState<boolean>(false);
  const pollRef = useRef<any>(null);

  const pollMetrics = () => {
    fetch("/api/train/metrics")
      .then((res) => res.json())
      .then((data) => {
        if (!data.success) return;
        setIsRunning(!!data.training_running);
        setCheckpointAvailable(!!data.checkpoint_available);
        if (Array.isArray(data.history)) {
          setMetrics(
            data.history.map((h: any) => ({
              epoch: h.epoch,
              step: h.step,
              loss: h.loss,
              perplexity: h.perplexity,
              learning_rate: h.learning_rate,
              timestamp: new Date().toLocaleTimeString(),
            }))
          );
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    pollMetrics();
    pollRef.current = setInterval(pollMetrics, 1500);
    return () => clearInterval(pollRef.current);
  }, []);

  const handleStart = () => {
    fetch("/api/train/start", { method: "POST" })
      .then(() => pollMetrics())
      .catch(() => {});
  };

  const handleStop = () => {
    fetch("/api/train/stop", { method: "POST" })
      .then(() => pollMetrics())
      .catch(() => {});
  };

  const handleReset = () => {
    fetch("/api/train/reset", { method: "POST" })
      .then(() => {
        setMetrics([]);
        pollMetrics();
      })
      .catch(() => {});
  };

  const latestMetric = metrics[metrics.length - 1];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <Code2 className="w-5 h-5 text-cyan-400" /> Training & Loss Workbench
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Real-time AdamW + Cosine Annealing Optimizer Telemetry (live from PyTorch backend)
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {!isRunning ? (
            <button
              onClick={handleStart}
              className="flex items-center space-x-1.5 bg-emerald-600 hover:bg-emerald-500 text-white px-3.5 py-2 rounded-xl text-xs font-semibold shadow-md transition-all"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Start Training</span>
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center space-x-1.5 bg-amber-600 hover:bg-amber-500 text-white px-3.5 py-2 rounded-xl text-xs font-semibold shadow-md transition-all"
            >
              <Square className="w-4 h-4 fill-current" />
              <span>Pause Training</span>
            </button>
          )}
          <button
            onClick={handleReset}
            className="flex items-center space-x-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 px-3.5 py-2 rounded-xl text-xs font-semibold border border-slate-700 transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-400 font-mono">Cross-Entropy Loss</span>
          <div className="text-2xl font-black text-cyan-400 font-mono mt-1">
            {latestMetric ? latestMetric.loss.toFixed(4) : "--"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-400 font-mono">Perplexity</span>
          <div className="text-2xl font-black text-blue-400 font-mono mt-1">
            {latestMetric ? latestMetric.perplexity.toFixed(2) : "--"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-400 font-mono">Total Steps</span>
          <div className="text-2xl font-black text-indigo-400 font-mono mt-1">
            {latestMetric ? latestMetric.step : "0"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-400 font-mono">Learning Rate</span>
          <div className="text-2xl font-black text-emerald-400 font-mono mt-1">
            {latestMetric ? latestMetric.learning_rate.toExponential(2) : "--"}
          </div>
        </div>
      </div>

      <div className="text-[11px] font-mono text-slate-500">
        Checkpoint available: <span className={checkpointAvailable ? "text-emerald-400" : "text-slate-500"}>{checkpointAvailable ? "yes" : "no"}</span>
      </div>

      {/* Real-time Loss Curve */}
      <div className="bg-slate-950 border border-slate-800 p-5 rounded-xl space-y-3">
        <div className="flex justify-between text-xs font-mono text-slate-400">
          <span>Loss Trajectory</span>
          <span>Target Loss: &lt; 0.20</span>
        </div>
        <div className="h-40 flex items-end gap-1.5 pt-4">
          {metrics.length > 0 ? (
            metrics.map((m, idx) => (
              <div
                key={idx}
                style={{ height: `${Math.min(100, (m.loss / 4.5) * 100)}%` }}
                className="flex-1 bg-gradient-to-t from-cyan-600 to-blue-400 rounded-t transition-all"
                title={`Step ${m.step}: Loss ${m.loss}`}
              ></div>
            ))
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-slate-500 font-mono">
              Click "Start Training" to observe real PyTorch gradient optimization steps.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
