import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { Documents } from "../src/Documents";
import * as service from "../src/documents-api";

const record: service.DocumentRecord = {
  doc_id: "doc_" + "a".repeat(64),
  filename: "report.pdf",
  status: "ready",
  managed: true,
  owns_vectors: true,
  size_bytes: 100,
  page_count: 2,
  chunk_count: 3,
  updated_at: "2026-01-01",
  error: null,
};
it("renders metadata and requires explicit deletion confirmation", async () => {
  const list = vi.spyOn(service, "listDocuments").mockResolvedValue([record]);
  const remove = vi
    .spyOn(service, "deleteDocument")
    .mockResolvedValue({ ...record, status: "deleted" });
  render(<Documents />);
  expect(await screen.findByText("report.pdf")).toBeInTheDocument();
  await userEvent.click(
    screen.getByRole("button", { name: "Delete report.pdf" }),
  );
  expect(screen.getByRole("alertdialog")).toBeInTheDocument();
  expect(remove).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Cancel deletion" }),
  );
  expect(remove).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Delete report.pdf" }),
  );
  list.mockResolvedValue([]);
  await userEvent.click(
    screen.getByRole("button", { name: "Confirm deletion" }),
  );
  await waitFor(() => expect(remove).toHaveBeenCalledWith(record.doc_id));
  expect(
    await screen.findByText("No indexed or managed documents found."),
  ).toBeInTheDocument();
});
it("uploads and refreshes inventory without equating transfer progress with processing", async () => {
  vi.spyOn(service, "listDocuments").mockResolvedValue([]);
  const transfer = vi
    .spyOn(service, "uploadPDF")
    .mockImplementation((_file, progress) => {
      progress(100);
      return {
        promise: Promise.resolve({ ...record, status: "queued" }),
        abort: vi.fn(),
      };
    });
  render(<Documents />);
  await userEvent.upload(
    screen.getByLabelText("Choose PDF"),
    new File(["%PDF-"], "report.pdf", { type: "application/pdf" }),
  );
  expect(transfer).toHaveBeenCalledOnce();
  expect(
    await screen.findByText(
      "Upload accepted. Processing status is shown below.",
    ),
  ).toBeInTheDocument();
});
it("rejects non-PDF drops before transfer", async () => {
  vi.spyOn(service, "listDocuments").mockResolvedValue([]);
  const transfer = vi.spyOn(service, "uploadPDF");
  render(<Documents />);
  fireEvent.drop(screen.getByRole("region", { name: "PDF upload" }), {
    dataTransfer: { files: [new File(["hello"], "bad.txt")] },
  });
  expect(await screen.findByRole("alert")).toHaveTextContent("non-empty PDF");
  expect(transfer).not.toHaveBeenCalled();
});
it("shows upload failures and does not show success", async () => {
  vi.spyOn(service, "listDocuments").mockResolvedValue([]);
  vi.spyOn(service, "uploadPDF").mockImplementation(() => ({
    promise: Promise.reject(new Error("Duplicate upload")),
    abort: vi.fn(),
  }));
  render(<Documents />);
  await userEvent.upload(
    screen.getByLabelText("Choose PDF"),
    new File(["%PDF-"], "report.pdf", { type: "application/pdf" }),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Duplicate upload",
  );
  expect(
    screen.queryByText("Upload accepted. Processing status is shown below."),
  ).not.toBeInTheDocument();
});
it("shows pending cleanup, recovery action, and read-only external documents", async () => {
  vi.spyOn(service, "listDocuments").mockResolvedValue([
    { ...record, status: "cleanup_required", error: "Cleanup required" },
    {
      ...record,
      doc_id: "external",
      filename: "external.pdf",
      managed: false,
      status: "external",
    },
  ]);
  const recover = vi.spyOn(service, "recoverDocuments").mockResolvedValue();
  render(<Documents />);
  await screen.findByText("external.pdf");
  expect(
    screen.queryByRole("button", { name: "Delete external.pdf" }),
  ).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Retry recovery" }));
  expect(recover).toHaveBeenCalledOnce();
});
