import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxy = Object.fromEntries(
    ["/health", "/auth", "/api/v1"].map((path) => [
      path,
      {
        target: env.API_PROXY_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    ]),
  );
  return {
    plugins: [react(), tailwindcss()],
    server: { proxy },
    preview: { proxy },
  };
});
