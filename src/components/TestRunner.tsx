import React, { useState } from "react";
import { CheckCircle2, Play, AlertCircle, Clock } from "lucide-react";
import { TestSuiteResponse } from "../types";

export const TestRunner: React.FC = () => {
  const [running, setRunning] = useState<boolean>(false);
  const [suiteResults, setSuiteResults] = useState<TestSuiteResponse | null>(null);

  const handleRunTests = () => {
    setRunning(true);
    fetch("/api/tests/run", { method: "POST" })
      .then((res) => res.json())
      .then((data) => {
        if (data.results) {
          setSuiteResults(data);
        }
      })
      .catch(() => {})
      .finally(() => setRunning(false));
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" /> Kodra AI Automated Test Suite
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Verification suite for KodraGPT model architecture, tokenizer, trainer, and inference
          </p>
        </div>

        <button
          onClick={handleRunTests}
          disabled={running}
          className="flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-xl text-xs font-semibold shadow-md transition-all"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>{running ? "Executing PyTest..." : "Run All Verification Tests"}</span>
        </button>
      </div>

      {!suiteResults && !running && (
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 text-center text-xs text-slate-500 font-mono">
          No test run yet. Click "Run All Verification Tests" to execute the real pytest suite.
        </div>
      )}

      {/* Summary Banner */}
      {suiteResults && (
        <div className="grid grid-cols-4 gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-center">
          <div>
            <span className="text-[11px] text-slate-500 uppercase block">Total Tests</span>
            <span className="text-xl font-bold text-white">{suiteResults.total}</span>
          </div>
          <div>
            <span className="text-[11px] text-slate-500 uppercase block">Passed</span>
            <span className="text-xl font-bold text-emerald-400">{suiteResults.passed}</span>
          </div>
          <div>
            <span className="text-[11px] text-slate-500 uppercase block">Failed</span>
            <span className="text-xl font-bold text-red-400">{suiteResults.failed}</span>
          </div>
          <div>
            <span className="text-[11px] text-slate-500 uppercase block">Skipped</span>
            <span className="text-xl font-bold text-amber-400">{suiteResults.skipped}</span>
          </div>
        </div>
      )}

      {/* Test Items Table */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden">
        <table className="w-full text-left font-mono text-xs">
          <thead className="bg-slate-900/80 border-b border-slate-800 text-slate-400">
            <tr>
              <th className="p-3">Test Case</th>
              <th className="p-3">Status</th>
              <th className="p-3">Duration</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {suiteResults?.results.map((t, idx) => (
              <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                <td className="p-3 text-slate-200 font-semibold">{t.name}</td>
                <td className="p-3">
                  {t.status === "PASSED" ? (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
                      <CheckCircle2 className="w-3 h-3" /> PASSED
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-red-950/80 text-red-400 border border-red-800/60">
                      <AlertCircle className="w-3 h-3" /> {t.status}
                    </span>
                  )}
                </td>
                <td className="p-3 text-slate-400">{t.duration_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
