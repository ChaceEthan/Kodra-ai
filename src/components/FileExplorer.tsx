import React, { useState, useEffect } from "react";
import { Folder, FileText, ChevronRight, ChevronDown, Code, CheckCircle, RefreshCw } from "lucide-react";

interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  children?: FileNode[];
  size?: number;
}

export const FileExplorer: React.FC = () => {
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>("README.md");
  const [fileContent, setFileContent] = useState<string>("");
  const [loadingContent, setLoadingContent] = useState<boolean>(false);
  const [expandedDirs, setExpandedDirs] = useState<Record<string, boolean>>({
    model: true,
    tokenizer: true,
    training: true,
    server: true,
  });

  const fetchTree = () => {
    fetch("/api/files")
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.tree) {
          setFileTree(data.tree);
        }
      })
      .catch((err) => console.error("Error fetching files:", err));
  };

  const fetchContent = (pathStr: string) => {
    setSelectedFile(pathStr);
    setLoadingContent(true);
    fetch(`/api/file-content?path=${encodeURIComponent(pathStr)}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setFileContent(data.content);
        } else {
          setFileContent(`// Error loading file: ${data.error}`);
        }
      })
      .catch((err) => setFileContent(`// Network error: ${err.message}`))
      .finally(() => setLoadingContent(false));
  };

  useEffect(() => {
    fetchTree();
    fetchContent("README.md");
  }, []);

  const toggleDir = (dirPath: string) => {
    setExpandedDirs((prev) => ({ ...prev, [dirPath]: !prev[dirPath] }));
  };

  const renderTree = (nodes: FileNode[]) => {
    return (
      <ul className="space-y-0.5 pl-2">
        {nodes.map((node) => {
          if (node.type === "directory") {
            const isExpanded = !!expandedDirs[node.path];
            return (
              <li key={node.path}>
                <button
                  onClick={() => toggleDir(node.path)}
                  className="w-full flex items-center space-x-1.5 px-2 py-1 rounded hover:bg-slate-800 text-xs font-mono text-slate-300 text-left"
                >
                  {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />}
                  <Folder className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  <span className="truncate">{node.name}</span>
                </button>
                {isExpanded && node.children && renderTree(node.children)}
              </li>
            );
          } else {
            const isSelected = selectedFile === node.path;
            return (
              <li key={node.path}>
                <button
                  onClick={() => fetchContent(node.path)}
                  className={`w-full flex items-center space-x-1.5 px-2 py-1 rounded text-xs font-mono text-left truncate transition-colors ${
                    isSelected ? "bg-cyan-500/20 text-cyan-300 font-semibold" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <FileText className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  <span className="truncate">{node.name}</span>
                </button>
              </li>
            );
          }
        })}
      </ul>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl grid grid-cols-1 md:grid-cols-4 min-h-[600px]">
      {/* File Tree Sidebar */}
      <div className="border-r border-slate-800 p-4 bg-slate-950/60 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Folder className="w-4 h-4 text-cyan-400" /> Kodra AI Codebase
            </span>
            <button
              onClick={fetchTree}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors"
              title="Refresh tree"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="max-h-[500px] overflow-y-auto no-scrollbar">
            {fileTree.length > 0 ? renderTree(fileTree) : <p className="text-xs text-slate-500 p-2">Loading directory structure...</p>}
          </div>
        </div>
        <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500 font-mono">
          Path: kodra-core/{selectedFile}
        </div>
      </div>

      {/* Code Viewer */}
      <div className="md:col-span-3 flex flex-col bg-slate-950">
        <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
          <span className="text-xs font-mono text-cyan-400 flex items-center gap-2">
            <Code className="w-4 h-4 text-cyan-400" />
            {selectedFile || "Select a file to view"}
          </span>
          <span className="text-[11px] text-slate-500 font-mono">Kodra AI Source Code</span>
        </div>

        <div className="flex-1 p-4 overflow-x-auto overflow-y-auto max-h-[550px]">
          {loadingContent ? (
            <div className="flex items-center justify-center h-full text-slate-500 font-mono text-xs">
              Loading source file...
            </div>
          ) : (
            <pre className="text-xs font-mono text-slate-200 leading-relaxed whitespace-pre font-normal">
              {fileContent}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
};
