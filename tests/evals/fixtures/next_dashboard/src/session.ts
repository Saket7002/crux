// Sessions are cookie-based; there is no token in local storage.
export type Session = { userId: string; orgId: string };

export function currentSession(): Session | null {
  return null;
}
