/**
 * API layer for Ma'at Legal AI Backend.
 * All endpoints require X-API-Key header for authentication.
 * Configuration via Vite environment variables (VITE_API_BASE, VITE_API_KEY)
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

const DEFAULT_TIMEOUT_MS = 30_000; // 30 seconds

const headers = (): HeadersInit => ({
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
});

/**
 * Fetch wrapper with timeout using AbortController.
 * Prevents hanging UI on backend stalls.
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

/* ── Types ── */

export interface SessionItem {
  session_id: string;
  preview: string;
  message_count: number;
}

export interface Message {
  type: "human" | "ai";
  content: string;
}

export interface ChatResponse {
  session_id: string;
  generation: string;
  law_domain: string;
}

/* ── Endpoints ── */

/** Start a new chat session */
export async function startSession(): Promise<{ session_id: string; message: string }> {
  const res = await fetchWithTimeout(`${API_BASE}/api/v1/chat/start`, {
    method: "POST",
    headers: headers(),
  });
  if (!res.ok) throw new Error(`Failed to start session: ${res.statusText}`);
  return res.json();
}

/** List all available chat sessions */
export async function getSessions(): Promise<SessionItem[]> {
  const res = await fetchWithTimeout(`${API_BASE}/api/v1/chat/sessions`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.statusText}`);
  const data = await res.json();
  return data.sessions;
}

/** Get chat history for a session */
export async function getHistory(sessionId: string): Promise<Message[]> {
  const res = await fetchWithTimeout(`${API_BASE}/api/v1/chat/${sessionId}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.statusText}`);
  const data = await res.json();
  return data.history;
}

/** Send a message to a session and get the AI response */
export async function sendMessage(sessionId: string, query: string): Promise<ChatResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/api/v1/chat/${sessionId}`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`Failed to send message: ${res.statusText}`);
  return res.json();
}

/** Delete a chat session */
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetchWithTimeout(`${API_BASE}/api/v1/chat/${sessionId}`, {
    method: "DELETE",
    headers: headers(),
  });
  if (!res.ok) throw new Error(`Failed to delete session: ${res.statusText}`);
}
