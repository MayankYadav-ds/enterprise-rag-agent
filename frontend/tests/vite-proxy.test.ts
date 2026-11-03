import { createServer as createHttpServer } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, expect, it } from "vitest";
import { createServer as createViteServer } from "vite";
import viteConfig from "../vite.config";

const open = (server: ReturnType<typeof createHttpServer>) =>
  new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

const close = (server: ReturnType<typeof createHttpServer>) =>
  new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );

it("forwards authentication requests and cookies through the configured Vite proxy", async () => {
  const forwarded: Array<{
    method: string;
    path: string;
    origin: string | undefined;
  }> = [];
  const backend = createHttpServer((request, response) => {
    forwarded.push({
      method: request.method || "",
      path: request.url || "",
      origin: request.headers.origin,
    });
    response.writeHead(200, {
      "content-type": "application/json",
      "set-cookie": "session=opaque; HttpOnly; SameSite=Lax; Path=/",
    });
    response.end(JSON.stringify({ forwarded: request.url }));
  });
  let vite: Awaited<ReturnType<typeof createViteServer>> | undefined;
  await open(backend);
  try {
    const backendAddress = backend.address() as AddressInfo;
    const generated = await viteConfig({
      mode: "test",
      command: "serve",
      isSsrBuild: false,
      isPreview: false,
    });
    const authProxy = generated.server?.proxy?.["/auth"];
    expect(authProxy).toBeTruthy();
    expect(generated.preview?.proxy).toHaveProperty("/auth");

    vite = await createViteServer({
      configFile: false,
      appType: "custom",
      server: {
        host: "127.0.0.1",
        port: 0,
        proxy: {
          ...generated.server?.proxy,
          "/auth": {
            ...(Array.isArray(authProxy) ? authProxy[0] : authProxy),
            target: `http://127.0.0.1:${backendAddress.port}`,
          },
        },
      },
    });
    await vite.listen();
    const viteAddress = vite.httpServer?.address() as AddressInfo;
    const origin = `http://127.0.0.1:${viteAddress.port}`;

    for (const [method, path] of [
      ["POST", "/auth/login"],
      ["GET", "/auth/me"],
      ["POST", "/auth/csrf"],
      ["POST", "/auth/logout"],
    ]) {
      const response = await fetch(origin + path, {
        method,
        headers: { Origin: origin },
      });
      expect(response.status).toBe(200);
      expect(response.headers.get("set-cookie")).toContain("HttpOnly");
      expect(await response.json()).toEqual({ forwarded: path });
    }

    expect(forwarded.map(({ path }) => path)).toEqual([
      "/auth/login",
      "/auth/me",
      "/auth/csrf",
      "/auth/logout",
    ]);
    expect(
      forwarded.every(({ origin: requestOrigin }) => requestOrigin === origin),
    ).toBe(true);
  } finally {
    await vite?.close();
    await close(backend);
  }
}, 30_000);
