
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,

    // 适配 WSL2 + Docker 的文件变更监听
    watch: {
      usePolling: true,
      interval: 1000,
    },

    // 前端 API 请求转发至 C++ Drogon
    proxy: {
      "/api": {
        target: process.env.SLM_API_PROXY_TARGET || "http://backend:8080",
        changeOrigin: true,
      },
    },
  },
});
