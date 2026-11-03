import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import App from "../src/App";
import {
  apiFetch,
  currentCsrfToken,
  setCsrfToken,
  AUTHENTICATION_EXPIRED_EVENT,
} from "../src/api";

function setup(
  options: {
    initiallyAuthenticated?: boolean;
    isAdmin?: boolean;
    failLogin?: boolean;
    protectedStatus?: number;
    documents?: unknown[];
  } = {},
) {
  let authenticated = Boolean(options.initiallyAuthenticated);
  const fetcher = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/auth/me")
        return new Response(
          JSON.stringify({
            id: "user-1",
            username: "alice",
            is_admin: options.isAdmin ?? true,
          }),
          { status: authenticated ? 200 : 401 },
        );
      if (path === "/auth/csrf")
        return new Response(JSON.stringify({ csrf_token: "fresh-csrf" }));
      if (path === "/auth/login")
        return options.failLogin
          ? new Response("{}", { status: 401 })
          : (() => {
              authenticated = true;
              return new Response(JSON.stringify({ csrf_token: "login-csrf" }));
            })();
      if (path === "/auth/logout")
        return new Response(JSON.stringify({ message: "Logged out" }));
      if (path.startsWith("/api/v1/") && options.protectedStatus !== undefined)
        return new Response("{}", { status: options.protectedStatus });
      if (path === "/health")
        return new Response(JSON.stringify({ status: "ok" }));
      if (path === "/api/v1/documents")
        return new Response(JSON.stringify(options.documents ?? []));
      return new Response("{}", { status: 200 });
    },
  );
  setCsrfToken(null);
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

it("restores an existing session and refreshes the in-memory CSRF token", async () => {
  const fetcher = setup({ initiallyAuthenticated: true });
  render(<App />);
  expect(await screen.findByText("alice")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Overview" })).toBeInTheDocument();
  expect(fetcher.mock.calls.some(([path]) => path === "/auth/csrf")).toBe(true);
});

it("shows a login error for invalid credentials and keeps workspace protected", async () => {
  setup({ failLogin: true });
  render(<App />);
  await userEvent.type(await screen.findByLabelText("Username"), "alice");
  await userEvent.type(screen.getByLabelText("Password"), "wrong-password");
  await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Invalid username or password",
  );
  expect(
    screen.queryByRole("button", { name: "Overview" }),
  ).not.toBeInTheDocument();
});

it("logs in successfully and displays the authenticated workspace", async () => {
  const fetcher = setup();
  render(<App />);
  await userEvent.type(await screen.findByLabelText("Username"), "alice");
  await userEvent.type(screen.getByLabelText("Password"), "correct-password");
  await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
  expect(
    await screen.findByRole("button", { name: "Overview" }),
  ).toBeInTheDocument();
  expect(fetcher.mock.calls.some(([path]) => path === "/auth/login")).toBe(
    true,
  );
  expect(currentCsrfToken()).toBe("login-csrf");
});

it("redirects an unauthenticated visit to the login screen", async () => {
  setup();
  render(<App />);
  expect(
    await screen.findByRole("heading", { name: "Welcome back" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "AI workspace" }),
  ).not.toBeInTheDocument();
});

it("logs out and clears the authenticated workspace", async () => {
  setup({ initiallyAuthenticated: true });
  render(<App />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Sign out" }),
  );
  expect(
    await screen.findByRole("heading", { name: "Welcome back" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Overview" }),
  ).not.toBeInTheDocument();
});

it("expires the UI and clears CSRF after concurrent protected-request 401s", async () => {
  setup({ initiallyAuthenticated: true, protectedStatus: 401 });
  render(<App />);
  await screen.findByRole("button", { name: "Sign out" });
  expect(currentCsrfToken()).toBe("fresh-csrf");
  const dispatch = vi.spyOn(window, "dispatchEvent");
  await Promise.all([
    apiFetch("/api/v1/query", { method: "POST" }),
    apiFetch("/api/v1/retrieve", { method: "POST" }),
  ]);
  expect(
    dispatch.mock.calls.filter(
      ([event]) => event.type === AUTHENTICATION_EXPIRED_EVENT,
    ),
  ).toHaveLength(1);
  expect(currentCsrfToken()).toBeNull();
  expect(
    await screen.findByRole("heading", { name: "Welcome back" }),
  ).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Sign out" }),
  ).not.toBeInTheDocument();
});

it("keeps the authenticated UI and CSRF token after a protected 403", async () => {
  setup({ initiallyAuthenticated: true, protectedStatus: 403 });
  render(<App />);
  await screen.findByRole("button", { name: "Sign out" });
  await userEvent.click(screen.getByRole("button", { name: "AI workspace" }));
  await userEvent.type(screen.getByLabelText("Ask a question"), "Revenue?");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 403");
  expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
  expect(currentCsrfToken()).toBe("fresh-csrf");
});

it("sends credentials and the in-memory CSRF header on state-changing requests", async () => {
  const fetcher = setup({ initiallyAuthenticated: true });
  render(<App />);
  await waitFor(() =>
    expect(fetcher.mock.calls.some(([path]) => path === "/auth/csrf")).toBe(
      true,
    ),
  );
  await userEvent.click(screen.getByRole("button", { name: "AI workspace" }));
  await userEvent.type(screen.getByLabelText("Ask a question"), "Revenue?");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  await waitFor(() =>
    expect(fetcher.mock.calls.some(([path]) => path === "/api/v1/query")).toBe(
      true,
    ),
  );
  const request = fetcher.mock.calls.find(([path]) => path === "/api/v1/query");
  expect(request?.[1]?.credentials).toBe("include");
  expect(new Headers(request?.[1]?.headers).get("X-CSRF-Token")).toBe(
    "fresh-csrf",
  );
});

it.each([
  { isAdmin: true, expected: true },
  { isAdmin: false, expected: false },
])(
  "shows document recovery only for admins (isAdmin=$isAdmin)",
  async ({ isAdmin, expected }) => {
    setup({
      initiallyAuthenticated: true,
      isAdmin,
      documents: [
        {
          doc_id: "doc_cleanup",
          filename: "cleanup.pdf",
          status: "cleanup_required",
          managed: true,
          owns_vectors: true,
          size_bytes: 100,
          page_count: 1,
          chunk_count: 1,
          updated_at: null,
          error: "Cleanup required",
        },
      ],
    });
    render(<App />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Documents" }),
    );
    expect(await screen.findByText("cleanup.pdf")).toBeInTheDocument();
    if (expected) {
      expect(
        screen.getByRole("button", { name: "Retry recovery" }),
      ).toBeInTheDocument();
    } else {
      expect(
        screen.queryByRole("button", { name: "Retry recovery" }),
      ).not.toBeInTheDocument();
    }
  },
);
