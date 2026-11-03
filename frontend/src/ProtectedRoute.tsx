import type { ReactNode } from "react";
import { useAuth } from "./AuthProvider";
import { Login } from "./Login";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading)
    return (
      <main className="login-page" role="status">
        Restoring session…
      </main>
    );
  if (!user) return <Login />;
  return children;
}
