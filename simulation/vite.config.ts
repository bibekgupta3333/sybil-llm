import { defineConfig } from "vitest/config";

// Relative base so `vite build` output opens from `vite preview` or any static host.
export default defineConfig({
  base: "./",
  server: { port: 5173, open: false },
  build: { target: "es2022", sourcemap: true },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
