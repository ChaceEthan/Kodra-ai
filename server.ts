import express from "express";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import { createServer as createViteServer } from "vite";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "10mb" }));

  const KODRA_DIR = path.join(process.cwd(), "kodra-core");

  // Spawn Python FastAPI backend process
  let pyProcess: any = null;
  function startFastAPI() {
    console.log("[Kodra AI Agent] Spawning Python FastAPI backend on port 8000...");
    const venvPython = process.platform === "win32"
      ? path.join(KODRA_DIR, ".venv", "Scripts", "python.exe")
      : path.join(KODRA_DIR, ".venv", "bin", "python");
    const pythonBin = fs.existsSync(venvPython) ? venvPython : (process.platform === "win32" ? "python" : "python3");
    pyProcess = spawn(pythonBin, ["-m", "uvicorn", "server.backend:app", "--host", "127.0.0.1", "--port", "8000"], {
      cwd: KODRA_DIR,
      env: { ...process.env, PYTHONPATH: KODRA_DIR },
    });

    pyProcess.stdout.on("data", (data: any) => {
      console.log(`[FastAPI stdout] ${data.toString().trim()}`);
    });

    pyProcess.stderr.on("data", (data: any) => {
      console.error(`[FastAPI stderr] ${data.toString().trim()}`);
    });

    pyProcess.on("exit", (code: number) => {
      console.log(`[FastAPI] Exited with code ${code}`);
    });
  }

  if (fs.existsSync(KODRA_DIR)) {
    startFastAPI();
  }

  // Helper to proxy requests to FastAPI backend
  async function proxyToPy(req: express.Request, res: express.Response) {
    try {
      const url = `http://127.0.0.1:8000${req.originalUrl}`;
      const options: RequestInit = {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
      };
      if (req.method !== "GET" && req.method !== "HEAD") {
        options.body = JSON.stringify(req.body);
      }
      const pyRes = await fetch(url, options);
      const data = await pyRes.json();
      res.status(pyRes.status).json(data);
    } catch (err: any) {
      res.status(503).json({
        success: false,
        error: `FastAPI backend unavailable: ${err.message}`,
        indicators: {
          backend_connected: false,
          model_initialized: false,
          dataset_loaded: false,
          training_running: false,
          checkpoint_available: false,
          inference_ready: false,
        },
      });
    }
  }

  // Helper to read directory recursively
  function getFileTree(dirPath: string, relativeDir: string = ""): any[] {
    if (!fs.existsSync(dirPath)) return [];
    const items = fs.readdirSync(dirPath, { withFileTypes: true });
    const result: any[] = [];

    for (const item of items) {
      if (item.name.startsWith(".") || item.name === "__pycache__" || item.name === "checkpoints") {
        continue;
      }
      const relPath = relativeDir ? `${relativeDir}/${item.name}` : item.name;
      const fullPath = path.join(dirPath, item.name);

      if (item.isDirectory()) {
        result.push({
          name: item.name,
          path: relPath,
          type: "directory",
          children: getFileTree(fullPath, relPath),
        });
      } else {
        const stats = fs.statSync(fullPath);
        result.push({
          name: item.name,
          path: relPath,
          type: "file",
          size: stats.size,
        });
      }
    }
    return result;
  }

  // --- API Endpoints ---

  // FastAPI Proxies
  app.get("/api/status", proxyToPy);
  app.get("/api/model-info", proxyToPy);
  app.post("/api/tokenize", proxyToPy);
  app.post("/api/generate", proxyToPy);
  app.post("/api/train/start", proxyToPy);
  app.post("/api/train/stop", proxyToPy);
  app.post("/api/train/reset", proxyToPy);
  app.get("/api/train/metrics", proxyToPy);
  app.post("/api/tests/run", proxyToPy);
  app.get("/api/attention", proxyToPy);

  // File tree & content
  app.get("/api/files", (req, res) => {
    try {
      const tree = getFileTree(KODRA_DIR);
      res.json({ success: true, tree });
    } catch (error: any) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  app.get("/api/file-content", (req, res) => {
    try {
      const filePath = req.query.path as string;
      if (!filePath) {
        return res.status(400).json({ success: false, error: "Path required" });
      }
      const fullPath = path.join(KODRA_DIR, filePath);
      if (!fullPath.startsWith(KODRA_DIR)) {
        return res.status(403).json({ success: false, error: "Access denied" });
      }
      if (!fs.existsSync(fullPath)) {
        return res.status(404).json({ success: false, error: "File not found" });
      }
      const content = fs.readFileSync(fullPath, "utf-8");
      res.json({ success: true, path: filePath, content });
    } catch (error: any) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Download Colab Notebook JSON
  app.get("/api/colab", (req, res) => {
    try {
      const colabPath = path.join(KODRA_DIR, "notebooks", "kodra_ai_training.ipynb");
      if (fs.existsSync(colabPath)) {
        const jsonContent = fs.readFileSync(colabPath, "utf-8");
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Content-Disposition", "attachment; filename=kodra_ai_training.ipynb");
        res.send(jsonContent);
      } else {
        res.status(404).json({ success: false, error: "Colab notebook not found" });
      }
    } catch (error: any) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Vite middleware in development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Kodra AI Agent] Server running on http://localhost:${PORT}`);
  });
}

startServer();
