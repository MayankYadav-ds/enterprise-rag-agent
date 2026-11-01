import { describe, expect, it, vi } from "vitest";
import { APIError, health, isQuery, query, retrieve } from "../src/api";
import { answer } from "./fixtures";

const body = {
  question: "What was revenue?",
  strategy: "hybrid" as const,
  top_k: 5,
  rerank: false,
};
const signal = () => new AbortController().signal;
describe("typed HTTP client", () => {
  it("uses actual query path and exact request schema", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(answer)));
    vi.stubGlobal("fetch", fetcher);
    expect(await query(body, signal())).toEqual(answer);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/query");
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual(body);
  });
  it("uses retrieval without generation", async () => {
    const result = {
      question: answer.question,
      strategy: answer.strategy,
      reranked: false,
      evidence: answer.evidence,
    };
    const fetcher = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(result)));
    vi.stubGlobal("fetch", fetcher);
    expect(await retrieve(body, signal())).toEqual(result);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/retrieve");
  });
  it.each([422, 502, 503, 500, 404])(
    "sanitizes HTTP %s failures",
    async (status) => {
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValue(new Response("PRIVATE_KEY=secret", { status })),
      );
      try {
        await query(body, signal());
        throw new Error("Expected rejection");
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        expect((error as APIError).status).toBe(status);
        expect((error as Error).message).not.toContain("secret");
      }
    },
  );
  it("rejects network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("private URL")));
    await expect(query(body, signal())).rejects.toThrow(
      "Cannot reach the backend",
    );
  });
  it("reports cancellation without claiming the server stopped", async () => {
    const controller = new AbortController();
    controller.abort();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new DOMException("Aborted", "AbortError")),
    );
    await expect(query(body, controller.signal)).rejects.toThrow(
      "server may still finish",
    );
  });
  it.each([
    {},
    { ...answer, status: "made_up" },
    { ...answer, evidence: [{}] },
    { ...answer, citations: [{}] },
  ])("rejects malformed response %#", async (value) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify(value))),
    );
    await expect(query(body, signal())).rejects.toThrow(
      "invalid answer response",
    );
  });
  it("rejects HTML on successful HTTP", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>proxy fallback</html>")),
    );
    await expect(query(body, signal())).rejects.toThrow("invalid response");
  });
  it("requires actual health shape", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response('{"status":"unknown"}')),
    );
    await expect(health(signal())).rejects.toThrow("Unexpected health");
  });
  it("does not accept nonfinite relevance", () => {
    expect(
      isQuery({
        ...answer,
        evidence: [{ ...answer.evidence[0], score: Infinity }],
      }),
    ).toBe(false);
  });
});
