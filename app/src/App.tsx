import { useState, useCallback, useEffect, useRef } from "react";
import SplashScreen from "./components/SplashScreen";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import ChatInput from "./components/ChatInput";
import {
  SessionItem,
  Message,
  startSession,
  getSessions,
  getHistory,
  sendMessage,
  deleteSession,
} from "./api";

type AppPhase = "splash" | "chat";

export default function App() {
  /* ── State ── */
  const [phase, setPhase] = useState<AppPhase>("splash");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lawDomain, setLawDomain] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Track previous message count to only auto-scroll on new messages
  const prevMessageCountRef = useRef(messages.length);

  /* ── Auto-scroll on NEW messages only ── */
  useEffect(() => {
    if (messages.length > prevMessageCountRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
    prevMessageCountRef.current = messages.length;
  }, [messages]);

  /* ── Load sessions on mount (after splash) ── */
  useEffect(() => {
    if (phase === "chat") {
      loadSessions();
    }
  }, [phase]);

  const loadSessions = async () => {
    try {
      const data = await getSessions();
      setSessions(data);
    } catch {
      // Backend might not be running — that's fine, sessions stay empty
      console.warn("Could not load sessions. Is the backend running?");
    }
  };

  /* ── Handlers ── */

  const handleSplashComplete = useCallback(() => {
    setPhase("chat");
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      const result = await startSession();
      setActiveSessionId(result.session_id);
      setMessages([]);
      setLawDomain(null);
      await loadSessions();
    } catch {
      // If backend is down, just reset locally
      setActiveSessionId(null);
      setMessages([]);
      setLawDomain(null);
    }
    setSidebarOpen(false);
  }, []);

  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setLawDomain(null);
    try {
      const history = await getHistory(sessionId);
      setMessages(history);
    } catch {
      setMessages([]);
    }
    setSidebarOpen(false);
  }, []);

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
        setLawDomain(null);
      }
      await loadSessions();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  }, [activeSessionId]);

  const handleSendMessage = useCallback(async (text: string) => {
    // Require an active session - no implicit creation
    const sessionId = activeSessionId;
    if (!sessionId) {
      // Show a temporary message in chat area to prompt session creation
      const promptMsg: Message = {
        type: "ai",
        content: "Please create a new chat session first using the 'New Chat' button in the sidebar.",
      };
      setMessages((prev) => [...prev, promptMsg]);
      return;
    }

    // Add user message to UI
    const userMsg: Message = { type: "human", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await sendMessage(sessionId, text);
      const aiMsg: Message = { type: "ai", content: response.generation };
      setMessages((prev) => [...prev, aiMsg]);
      setLawDomain(response.law_domain);
      // Refresh session list to update previews
      await loadSessions();
    } catch (err) {
      const errorMsg: Message = {
        type: "ai",
        content: "Sorry, I encountered an error processing your request. Please make sure the backend server is running.",
      };
      setMessages((prev) => [...prev, errorMsg]);
      console.error("Send message error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [activeSessionId]);

  const handleSuggestionClick = useCallback((text: string) => {
    handleSendMessage(text);
  }, [handleSendMessage]);

  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  /* ── Render ── */
  return (
    <>
      {phase === "splash" && (
        <SplashScreen onComplete={handleSplashComplete} />
      )}

      <div className="app-layout" id="app-layout">
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          sidebarOpen={sidebarOpen}
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
          onDeleteSession={handleDeleteSession}
          onToggleSidebar={handleToggleSidebar}
        />

        <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          <ChatArea
            messages={messages}
            isLoading={isLoading}
            onSuggestionClick={handleSuggestionClick}
            lawDomain={lawDomain}
          />
          <ChatInput onSend={handleSendMessage} disabled={isLoading} hasSession={!!activeSessionId} />
          {/* Invisible anchor for auto-scroll */}
          <div ref={messagesEndRef} />
        </main>
      </div>
    </>
  );
}
