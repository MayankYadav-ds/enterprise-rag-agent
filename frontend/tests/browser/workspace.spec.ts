import { expect, test } from "@playwright/test";
import { answer } from "../fixtures";

test("upload, live inventory, confirmation and document-specific deletion", async ({
  page,
}, testInfo) => {
  const record = {
    doc_id: "doc_" + "a".repeat(64),
    filename: "sample.pdf",
    status: "ready",
    managed: true,
    owns_vectors: true,
    size_bytes: 100,
    page_count: 1,
    chunk_count: 2,
    updated_at: "2026-01-01T00:00:00Z",
    error: null,
  };
  let uploaded = false;
  let deleted = false;
  await page.route("**/health", (route) =>
    route.fulfill({ json: { status: "ok" } }),
  );
  await page.route("**/api/v1/documents**", (route) => {
    const request = route.request();
    if (request.method() === "POST") {
      expect(request.headers()["content-type"]).toBe("application/pdf");
      expect(request.postDataBuffer()?.toString()).toBe("%PDF-test");
      uploaded = true;
      return route.fulfill({
        status: 202,
        json: { ...record, status: "queued" },
      });
    }
    if (request.method() === "DELETE") {
      expect(request.url()).toContain(record.doc_id);
      deleted = true;
      return route.fulfill({ json: { ...record, status: "deleted" } });
    }
    return route.fulfill({ json: uploaded && !deleted ? [record] : [] });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Documents", exact: true }).click();
  await page.getByLabel("Choose PDF").setInputFiles({
    name: "sample.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-test"),
  });
  await expect(page.getByText("ready", { exact: true })).toBeVisible();
  await expect(page.getByText("sample.pdf", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete sample.pdf" }).click();
  await expect(page.getByRole("alertdialog")).toBeVisible();
  expect(deleted).toBe(false);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("document-management.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Confirm deletion" }).click();
  await expect(
    page.getByText("No indexed or managed documents found."),
  ).toBeVisible();
  expect(deleted).toBe(true);
});

test("all screens, responsive layout, settings, answer and source inspection", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/health", (route) =>
    route.fulfill({ json: { status: "ok" } }),
  );
  await page.route("**/api/v1/query", (route) => {
    expect(route.request().postDataJSON()).toEqual({
      question: "What was revenue?",
      strategy: "bm25",
      top_k: 3,
      rerank: true,
    });
    return route.fulfill({
      json: { ...answer, strategy: "bm25", reranked: true },
    });
  });
  await page.route("**/api/v1/documents", (route) =>
    route.fulfill({ json: [] }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Find the answer/ }),
  ).toBeVisible();
  const noOverflow = async () =>
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  await noOverflow();
  await page.screenshot({
    path: testInfo.outputPath("overview.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Documents", exact: true }).click();
  await expect(page.getByLabel("Choose PDF")).toBeEnabled();
  await noOverflow();
  await page.screenshot({
    path: testInfo.outputPath("documents.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "AI workspace", exact: true }).click();
  await page.getByRole("button", { name: "Retrieval settings" }).click();
  await page.getByLabel("Retrieval strategy").selectOption("bm25");
  await page.getByLabel("Top-k passages").fill("3");
  await page.getByLabel("Cross-encoder reranking").check();
  await page.getByLabel("Ask a question").fill("What was revenue?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByText("Evidence-grounded answer")).toBeVisible();
  await page.locator(".citations summary").click();
  await expect(page.locator(".citations blockquote")).toHaveText(
    "Revenue was $10 million.",
  );
  await page.locator(".evidence-card summary").click();
  await expect(page.locator(".passage-body")).toContainText(
    "Revenue was $10 million.",
  );
  await noOverflow();
  await page.screenshot({
    path: testInfo.outputPath("workspace.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByRole("button", { name: "AI workspace", exact: true }).click();
  await expect(page.getByText("Evidence-grounded answer")).toBeVisible();
  expect(errors).toEqual([]);
});

test("provider failure is visible and does not render a fabricated answer", async ({
  page,
}) => {
  await page.route("**/health", (route) => route.abort());
  await page.route("**/api/v1/query", (route) =>
    route.fulfill({ status: 503, json: { code: "dependency_unavailable" } }),
  );
  await page.route("**/api/v1/documents", (route) =>
    route.fulfill({ json: [] }),
  );
  await page.goto("/");
  await expect(page.getByText("Backend unavailable").first()).toBeVisible();
  await page.getByRole("button", { name: "AI workspace", exact: true }).click();
  await page.getByLabel("Ask a question").fill("Revenue?");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(
    "required service or model is unavailable",
  );
  await expect(page.getByText("Evidence-grounded answer")).toHaveCount(0);
});
