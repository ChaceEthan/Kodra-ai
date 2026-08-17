import React from "react";
import { Cpu, Code2, Sparkles, Terminal, FileCode, CheckCircle2 } from "lucide-react";
import { SystemStatusIndicators } from "../types";

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  paramCount: string;
  indicators?: SystemStatusIndicators;
  device?: string;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, paramCount, indicators, device }) => {
  const tabs = [
    { id: "overview", label: "Model Architecture", icon: Cpu },
    { id: "explorer", label: "Codebase Explorer", icon: FileCode },
    { id: "tokenizer", label: "Tokenizer Workbench", icon: Terminal },
    { id: "attention", label: "Causal Attention", icon: Sparkles },
    { id: "training", label: "Training & Loss", icon: Code2 },
    { id: "generation", label: "Code Completion", icon: Code2 },
    { id: "tests", label: "Unit Test Suite", icon: CheckCircle2 },
  ];

  const defaultIndicators: SystemStatusIndicators = {
    backend_connected: true,
    model_initialized: true,
    dataset_loaded: true,
    training_running: false,
    checkpoint_available: true,
    inference_ready: true,
  };

  const currentIndicators = indicators || defaultIndicators;

  const statusItems = [
    { key: "backend_connected", label: "Backend connected", active: currentIndicators.backend_connected },
    { key: "model_initialized", label: "Model initialized", active: currentIndicators.model_initialized },
    { key: "dataset_loaded", label: "Dataset loaded", active: currentIndicators.dataset_loaded },
    { key: "training_running", label: "Training running", active: currentIndicators.training_running },
    { key: "checkpoint_available", label: "Checkpoint available", active: currentIndicators.checkpoint_available },
    { key: "inference_ready", label: "Inference ready", active: currentIndicators.inference_ready },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center p-1.5 bg-slate-950 rounded-xl border border-slate-800 shadow-md">
              <img
                src="/public/branding/kodra_logo_icon.svg"
                alt="Kodra AI Logo"
                className="w-8 h-8 object-contain"
                onError={(e) => {
                  // Fallback icon if image error occurs
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-black tracking-tight text-white flex items-center gap-1.5">
                  <span>KODRA</span>
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">AI AGENT</span>
                </h1>
                <span className="bg-cyan-950 text-cyan-400 border border-cyan-800/60 text-xs font-semibold px-2 py-0.5 rounded-full">
                  Phase 1 PyTorch Core
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Scratch PyTorch Causal GPT Code Model • {paramCount} Parameters
              </p>
            </div>
          </div>

          {/* Tagline & Actions */}
          <div className="hidden lg:flex items-center space-x-4">
            <div className="text-right">
              <span className="text-[11px] font-bold tracking-widest text-cyan-400 uppercase block">CODE • THINK • CREATE</span>
              <span className="text-xs text-slate-400">Your Intelligent Coding Partner</span>
            </div>
            <div className="h-8 w-px bg-slate-800"></div>
            <div className="flex items-center space-x-2 bg-slate-800/80 border border-slate-700/60 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>{device || "CPU (PyTorch 2.13)"}</span>
            </div>
            <a
              href="/api/colab"
              download="kodra_ai_training.ipynb"
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-3 py-1.5 rounded-lg text-xs font-semibold transition-all shadow-md shadow-cyan-950"
            >
              <span>Download Colab .ipynb</span>
            </a>
          </div>
        </div>

        {/* Status Indicators Bar */}
        <div className="flex items-center space-x-2 overflow-x-auto py-1.5 border-t border-slate-800/80 text-[11px] font-mono no-scrollbar">
          {statusItems.map((item) => (
            <div
              key={item.key}
              className={`flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full border text-[10px] font-semibold whitespace-nowrap transition-colors ${
                item.active
                  ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                  : "bg-slate-950/60 text-slate-500 border-slate-800/60"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${item.active ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`}></span>
              <span>{item.label}</span>
            </div>
          ))}
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 overflow-x-auto no-scrollbar py-2 border-t border-slate-800/60">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
