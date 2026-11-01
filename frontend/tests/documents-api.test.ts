import { expect, it, vi } from "vitest";
import { deleteDocument, listDocuments, uploadPDF } from "../src/documents-api";

it("rejects malformed inventory rather than showing an empty library", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}")));
  await expect(listDocuments()).rejects.toThrow("Invalid document inventory");
});
it("uses scoped DELETE and does not trust unconfirmed deletion", async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetcher);
  await expect(deleteDocument("doc_abc")).rejects.toThrow(
    "Deletion was not confirmed",
  );
  expect(fetcher.mock.calls[0][0]).toBe("/api/v1/documents/doc_abc");
  expect(fetcher.mock.calls[0][1].method).toBe("DELETE");
});
it.each([409, 413, 415, 422, 503])(
  "handles upload HTTP %s without leaking server details",
  async (status) => {
    class XHR {
      upload = {};
      status = status;
      responseText = "secret exception";
      timeout = 0;
      onload = () => {};
      open = vi.fn();
      setRequestHeader = vi.fn();
      abort = vi.fn();
      send() {
        queueMicrotask(() => this.onload());
      }
    }
    vi.stubGlobal("XMLHttpRequest", XHR);
    await expect(
      uploadPDF(new File(["x"], "a.pdf"), vi.fn()).promise,
    ).rejects.not.toThrow("secret");
  },
);
