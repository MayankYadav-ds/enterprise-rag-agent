import { APIError, apiFetch, setCsrfToken } from "../api";

export interface AuthUser {
  id: string;
  username: string;
  is_admin: boolean;
}

async function json(response: Response) {
  if (!response.ok) {
    const message =
      response.status === 401
        ? "Invalid username or password."
        : `Authentication request failed (HTTP ${response.status}).`;
    throw new APIError(response.status, message);
  }
  try {
    return (await response.json()) as unknown;
  } catch {
    throw new APIError(502, "Invalid authentication response.");
  }
}

function user(value: unknown): value is AuthUser {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.id === "string" &&
    typeof record.username === "string" &&
    typeof record.is_admin === "boolean"
  );
}

async function obtainCsrf(): Promise<void> {
  const data = await json(await apiFetch("/auth/csrf", { method: "POST" }));
  if (
    !data ||
    typeof data !== "object" ||
    typeof (data as Record<string, unknown>).csrf_token !== "string"
  )
    throw new APIError(502, "Invalid CSRF response.");
  setCsrfToken((data as { csrf_token: string }).csrf_token);
}

export async function currentUser(): Promise<AuthUser> {
  const data = await json(await apiFetch("/auth/me"));
  if (!user(data)) throw new APIError(502, "Invalid user response.");
  await obtainCsrf();
  return data;
}

export async function login(
  username: string,
  password: string,
): Promise<AuthUser> {
  const data = await json(
    await apiFetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }),
  );
  if (
    !data ||
    typeof data !== "object" ||
    typeof (data as Record<string, unknown>).csrf_token !== "string"
  )
    throw new APIError(502, "Invalid login response.");
  setCsrfToken((data as { csrf_token: string }).csrf_token);
  const me = await json(await apiFetch("/auth/me"));
  if (!user(me)) throw new APIError(502, "Invalid user response.");
  return me;
}

export async function logout(): Promise<void> {
  try {
    await json(await apiFetch("/auth/logout", { method: "POST" }));
  } finally {
    setCsrfToken(null);
  }
}
