import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// While developing ("npm run dev") the page runs on its own port, and every request that starts with /api is
// passed on to the Python server, so the page can just call "/api/..." in both development and the built version.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
