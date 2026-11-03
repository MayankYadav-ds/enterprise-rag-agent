import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import * as auth from "./api/auth";
import { AUTHENTICATION_EXPIRED_EVENT, setCsrfToken } from "./api";

interface AuthContextValue {
  user: auth.AuthUser | null;
  loading: boolean;
  error: string;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<auth.AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void auth
      .currentUser()
      .then((current) => {
        if (active) setUser(current);
      })
      .catch(() => {
        if (active) setUser(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const expireSession = () => {
      setCsrfToken(null);
      setUser(null);
      setError("Your session has expired. Please sign in again.");
      setLoading(false);
    };
    window.addEventListener(AUTHENTICATION_EXPIRED_EVENT, expireSession);
    return () =>
      window.removeEventListener(AUTHENTICATION_EXPIRED_EVENT, expireSession);
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    setError("");
    try {
      setUser(await auth.login(username, password));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Login failed.");
      throw cause;
    }
  }, []);

  const logout = useCallback(async () => {
    await auth.logout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, error, login, logout }),
    [user, loading, error, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used within AuthProvider");
  return value;
}
