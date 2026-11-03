import { useState, type FormEvent } from "react";
import { Layers3, LogIn } from "lucide-react";
import { useAuth } from "./AuthProvider";

export function Login() {
  const { login, error } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setPending(true);
    try {
      await login(username, password);
    } catch {
      // The provider exposes a generic, user-safe error.
    } finally {
      setPending(false);
    }
  };
  return (
    <main className="login-page">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-icon">
          <Layers3 size={24} />
        </div>
        <div className="eyebrow">PRIVATE DOCUMENT WORKSPACE</div>
        <h1>Welcome back</h1>
        <p>Sign in to access your evidence workspace.</p>
        <label htmlFor="username">Username</label>
        <input
          id="username"
          autoComplete="username"
          required
          maxLength={50}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
        />
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          required
          maxLength={1024}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error && (
          <p role="alert" className="error-message">
            {error}
          </p>
        )}
        <button
          className="primary"
          type="submit"
          disabled={pending || !username || !password}
        >
          <LogIn size={17} /> {pending ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
