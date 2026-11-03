import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import App from "../src/App";
import { setCsrfToken } from "../src/api";
import { answer } from "./fixtures";

function mockAPI(result: unknown = answer, status = 200) {
  const fetcher = vi.fn(async (path: string) => {
    const body =
      path === "/health"
        ? { status: "ok" }
        : path === "/auth/me"
          ? { id: "u1", username: "alice", is_admin: true }
          : path === "/auth/csrf" || path === "/auth/login"
            ? { csrf_token: "csrf-test-token" }
            : path === "/auth/logout"
              ? { message: "Logged out" }
              : path === "/api/v1/documents"
                ? []
                : result;
    const responseStatus = path.startsWith("/api/v1/") ? status : 200;
    return new Response(JSON.stringify(body), { status: responseStatus });
  });
  setCsrfToken(null);
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}
async function openChat() {
  await userEvent.click(
    await screen.findByRole("button", { name: "AI workspace" }),
  );
}
async function send() {
  await userEvent.type(
    screen.getByLabelText("Ask a question"),
    "What was revenue?",
  );
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}
it("shows liveness and honest unavailable corpus statistics", async () => {
  mockAPI();
  render(<App />);
  await waitFor(() =>
    expect(screen.getAllByText("Backend online")).toHaveLength(2),
  );
  expect(
    screen.getByText("Open Documents for the live inventory"),
  ).toBeInTheDocument();
  expect(screen.getByText(/Liveness only/)).toBeInTheDocument();
});
it("opens document upload and the real inventory endpoint", async () => {
  const fetcher = mockAPI();
  render(<App />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Documents" }),
  );
  expect(screen.getByLabelText("Choose PDF")).toBeEnabled();
  expect(
    await screen.findByText("No indexed or managed documents found."),
  ).toBeInTheDocument();
  expect(
    fetcher.mock.calls.some(([path]) => path === "/api/v1/documents"),
  ).toBe(true);
});
it("shows grounded answer, actual page ranges, citations and expandable evidence", async () => {
  mockAPI();
  render(<App />);
  await openChat();
  await send();
  expect(
    await screen.findByText("Evidence-grounded answer"),
  ).toBeInTheDocument();
  expect(screen.getAllByText("report.pdf")).toHaveLength(2);
  expect(screen.getByText("Pages 7–8")).toBeInTheDocument();
  expect(screen.getByText(/raw score 0.8200/)).toBeInTheDocument();
  expect(screen.getByText(/not confidence percentages/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Overview" }));
  await openChat();
  expect(screen.getByText("Evidence-grounded answer")).toBeInTheDocument();
});
it.each(["insufficient_evidence", "conflicting_evidence"] as const)(
  "preserves %s as an explicit status",
  async (status) => {
    mockAPI({
      ...answer,
      status,
      answer: "Review the available evidence.",
      citations: [],
      evidence: [],
    });
    render(<App />);
    await openChat();
    await send();
    expect(
      await screen.findByText(
        status === "insufficient_evidence"
          ? "Insufficient evidence"
          : "Conflicting evidence",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("No passages returned")).toBeInTheDocument();
  },
);
it.each([422, 502, 503])(
  "shows HTTP %s error without a successful answer",
  async (status) => {
    mockAPI({ message: "PRIVATE_KEY" }, status);
    render(<App />);
    await openChat();
    await send();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Request not completed",
    );
    expect(
      screen.queryByText("Evidence-grounded answer"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("PRIVATE_KEY")).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "Edit and try again" }),
    );
    expect(screen.getByLabelText("Ask a question")).toHaveValue(
      "What was revenue?",
    );
  },
);
it("passes retrieval settings and uses retrieve endpoint in retrieval-only mode", async () => {
  const result = {
    question: answer.question,
    strategy: "bm25",
    reranked: true,
    evidence: answer.evidence,
  };
  const fetcher = mockAPI(result);
  render(<App />);
  await openChat();
  await userEvent.click(
    screen.getByRole("button", { name: "Retrieval settings" }),
  );
  await userEvent.selectOptions(
    screen.getByLabelText("Retrieval strategy"),
    "bm25",
  );
  fireEvent.change(screen.getByLabelText("Top-k passages"), {
    target: { value: "3" },
  });
  await userEvent.click(screen.getByLabelText("Cross-encoder reranking"));
  await userEvent.selectOptions(
    screen.getByLabelText("Response mode"),
    "retrieve",
  );
  await send();
  expect(await screen.findByText("Retrieval complete")).toBeInTheDocument();
  const call = fetcher.mock.calls.find(([path]) => path === "/api/v1/retrieve");
  expect(call).toBeTruthy();
  expect(fetcher.mock.calls.some(([path]) => path === "/api/v1/query")).toBe(
    false,
  );
});
it("prevents blank questions and out-of-range top-k", async () => {
  mockAPI();
  render(<App />);
  await openChat();
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  await userEvent.type(screen.getByLabelText("Ask a question"), "Revenue?");
  await userEvent.click(
    screen.getByRole("button", { name: "Retrieval settings" }),
  );
  fireEvent.change(screen.getByLabelText("Top-k passages"), {
    target: { value: "101" },
  });
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
});
it("shows backend unavailable on network error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      if (path === "/auth/me")
        return new Response(
          JSON.stringify({ id: "u1", username: "alice", is_admin: true }),
        );
      if (path === "/auth/csrf")
        return new Response(JSON.stringify({ csrf_token: "csrf-test-token" }));
      throw new TypeError("Failed to fetch");
    }),
  );
  render(<App />);
  await waitFor(() =>
    expect(screen.getAllByText("Backend unavailable")).toHaveLength(2),
  );
});
it("renders model strings as text rather than executable HTML", async () => {
  mockAPI({ ...answer, answer: "<img src=x onerror=alert(1)>" });
  const { container } = render(<App />);
  await openChat();
  await send();
  expect(
    await screen.findByText("<img src=x onerror=alert(1)>"),
  ).toBeInTheDocument();
  expect(container.querySelector("img")).toBeNull();
});
