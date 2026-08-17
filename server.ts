import "dotenv/config";
import express from "express";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import { createServer as createViteServer } from "vite";
import { resolveWithinRoot } from "./server-utils";

// --- Environment-driven configuration (safe local-dev defaults) ------------
const APP_HOST = process.env.APP_HOST || "0.0.0.0";
const APP_PORT = parseInt(process.env.APP_PORT || "3000", 10);

const KODRA_BACKEND_HOST = process.env.KODRA_BACKEND_HOST || "127.0.0.1";
const KODRA_BACKEND_PORT = parseInt(process.env.KODRA_BACKEND_PORT || "8000", 10);
const KODRA_BACKEND_URL =
  process.env.KODRA_BACKEND_URL || `http://${KODRA_BACKEND_HOST}:${KODRA_BACKEND_PORT}`;

const KODRA_CORE_DIR = path.resolve(process.cwd(), process.env.KODRA_CORE_DIR || "./kodra-core");

const PROXY_TIMEOUT_MS = parseInt(process.env.KODRA_PROXY_TIMEOUT_MS || "15000", 10);

const ALLOWED_ORIGINS = (process.env.KODRA_ALLOWED_ORIGINS || "http://localhost:3000")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);

async function startServer() {
  const app = express();

  app.use(express.json({ limit: "10mb" }));

  // Minimal same-origin-friendly CORS guard for the gateway itself. The
  // frontend is normally served by this same process, so this mostly
  // matters when the dev server is hit cross-origin (e.g. a separate Vite
  // preview) or when the gateway is used as a pure API by another client.
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && ALLOWED_ORIGINS.includes(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Access-Control-Allow-Credentials", "true");
    }
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });

  // Spawn Python FastAPI backend process
  let pyProcess: any = null;
  function startFastAPI() {
    console.log(
      `[Kodra AI Agent] Spawning Python FastAPI backend on ${KODRA_BACKEND_HOST}:${KODRA_BACKEND_PORT}...`
    );
    const venvPython =
      process.platform === "win32"
        ? path.join(KODRA_CORE_DIR, ".venv", "Scripts", "python.exe")
        : path.join(KODRA_CORE_DIR, ".venv", "bin", "python");
    const pythonBin = fs.existsSync(venvPython)
      ? venvPython
      : process.platform === "win32"
      ? "python"
      : "python3";
    pyProcess = spawn(
      pythonBin,
      ["-m", "uvicorn", "server.backend:app", "--host", KODRA_BACKEND_HOST, "--port", String(KODRA_BACKEND_PORT)],
      {
        cwd: KODRA_CORE_DIR,
        env: { ...process.env, PYTHONPATH: KODRA_CORE_DIR },
      }
    );

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

  if (fs.existsSync(KODRA_CORE_DIR)) {
    startFastAPI();
  } else {
    console.warn(`[Kodra AI Agent] KODRA_CORE_DIR not found at ${KODRA_CORE_DIR} - backend not started.`);
  }

  // Helper to proxy requests to FastAPI backend with a configurable timeout
  // and correct upstream-status/error propagation.
  async function proxyToPy(req: express.Request, res: express.Response) {
    const url = `${KODRA_BACKEND_URL}${req.originalUrl}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROXY_TIMEOUT_MS);

    try {
      const options: RequestInit = {
        method: req.method,
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      };
      if (req.method !== "GET" && req.method !== "HEAD") {
        options.body = JSON.stringify(req.body);
      }
      const pyRes = await fetch(url, options);
      const text = await pyRes.text();

      let data: any;
      try {
        data = JSON.parse(text);
      } catch {
        // FastAPI/uvicorn returned a non-JSON body (e.g. an HTML error page
        // or plain-text traceback) - surface it without leaking internals.
        data = { success: false, error: "Backend returned a non-JSON response" };
      }
      res.status(pyRes.status).json(data);
    } catch (err: any) {
      const timedOut = err?.name === "AbortError";
      console.error(`[Kodra AI Agent] Proxy error for ${req.originalUrl}: ${timedOut ? "timeout" : err?.message}`);
      res.status(503).json({
        success: false,
        error: timedOut ? `Backend request timed out after ${PROXY_TIMEOUT_MS}ms` : "Backend unavailable",
        indicators: {
          backend_connected: false,
          model_initialized: false,
          trained_checkpoint_loaded: false,
          dataset_loaded: false,
          training_running: false,
          checkpoint_available: false,
          inference_available: false,
        },
      });
    } finally {
      clearTimeout(timer);
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
  app.get("/api/backend-health", proxyToPy); // proxies to FastAPI GET /api/health

  // Gateway-level health check: real state only, no fabricated fields.
  app.get("/api/health", async (req, res) => {
    let backendReachable = false;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);
      const r = await fetch(`${KODRA_BACKEND_URL}/api/health`, { signal: controller.signal });
      clearTimeout(timer);
      backendReachable = r.ok;
    } catch {
      backendReachable = false;
    }
    res.json({
      success: true,
      gateway: "ok",
      backend_reachable: backendReachable,
      backend_url: KODRA_BACKEND_URL,
    });
  });

  // File tree & content
  app.get("/api/files", (req, res) => {
    try {
      const tree = getFileTree(KODRA_CORE_DIR);
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
      const fullPath = resolveWithinRoot(KODRA_CORE_DIR, filePath);
      if (fullPath === null) {
        return res.status(403).json({ success: false, error: "Access denied" });
      }
      if (!fs.existsSync(fullPath)) {
        return res.status(404).json({ success: false, error: "File not found" });
      }
      if (!fs.statSync(fullPath).isFile()) {
        return res.status(400).json({ success: false, error: "Path is not a regular file" });
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
      const colabPath = path.join(KODRA_CORE_DIR, "notebooks", "kodra_ai_training.ipynb");
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

  app.listen(APP_PORT, APP_HOST, () => {
    console.log(`[Kodra AI Agent] Server running on http://localhost:${APP_PORT}`);
  });
}

startServer();

export { };
