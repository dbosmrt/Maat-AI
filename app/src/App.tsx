import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import SplashScreen from "./components/SplashScreen";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import ChatInput from "./components/ChatInput";
import { ToastProvider, useToast } from "./hooks/useToast";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
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

function AppContent() {
  /* ── State ── */
  const [phase, setPhase] = useState<AppPhase>("splash");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lawDomain, setLawDomain] = useState<string | null>(null);

  const { showToast } = useToast();

  const messagesEndRef = useRef<HTMLDivElement>(null);
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

  const loadSessions = useCallback(async () => {
    try {
      const data = await getSessions();
      setSessions(data);
    } catch {
      showToast({
        type: "warning",
        title: "Backend unavailable",
        message: "Could not load sessions. Is the backend running?",
        duration: 0,
        action: {
          label: "Retry",
          onClick: loadSessions,
        },
      });
      console.warn("Could not load sessions. Is the backend running?");
    }
  }, [showToast]);

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
      showToast({ type: "success", title: "New chat started" });
    } catch {
      setActiveSessionId(null);
      setMessages([]);
      setLawDomain(null);
      showToast({ type: "error", title: "Failed to start chat", message: "Backend may be unavailable" });
    }
    setSidebarOpen(false);
  }, [loadSessions, showToast]);

  /* ── Keyboard shortcuts ── */
  const shortcuts = useMemo(
    () => [
      {
        key: "n",
        ctrl: true,
        action: handleNewChat,
        description: "New chat",
      },
      {
        key: "/",
        ctrl: true,
        action: () => setSidebarOpen(true),
        description: "Focus search",
      },
      {
        key: "Escape",
        action: () => setSidebarOpen(false),
        description: "Close sidebar",
      },
    ],
    [handleNewChat]
  );

  useKeyboardShortcuts(shortcuts);

  const handleSelectSession = useCallback(async (sessionId: string) => {
    setActiveSessionId(sessionId);
    setLawDomain(null);
    try {
      const history = await getHistory(sessionId);
      setMessages(history);
    } catch {
      setMessages([]);
      showToast({ type: "error", title: "Failed to load history" });
    }
    setSidebarOpen(false);
  }, [showToast]);

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
        setLawDomain(null);
      }
      await loadSessions();
      showToast({ type: "success", title: "Conversation deleted" });
    } catch (err) {
      console.error("Failed to delete session:", err);
      showToast({ type: "error", title: "Failed to delete" });
    }
  }, [activeSessionId, loadSessions, showToast]);

  const handleSendMessage = useCallback(async (text: string) => {
    // Require an active session - no implicit creation
    const sessionId = activeSessionId;
    if (!sessionId) {
      showToast({
        type: "warning",
        title: "No active chat",
        message: "Create a new chat session first using the sidebar",
        action: { label: "New Chat", onClick: handleNewChat },
      });
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
      await loadSessions();
    } catch (err) {
      const errorMsg: Message = {
        type: "ai",
        content: "Sorry, I encountered an error processing your request. Please make sure the backend server is running.",
        isError: true,
        errorMessage: err instanceof Error ? err.message : "Unknown error",
      };
      setMessages((prev) => [...prev, errorMsg]);
      console.error("Send message error:", err);
      showToast({ type: "error", title: "Error sending message", message: "Please try again" });
    } finally {
      setIsLoading(false);
    }
  }, [activeSessionId, handleNewChat, loadSessions, showToast]);

  const handleRegenerate = useCallback(async (messageIndex: number) => {
    // Find the user message before this AI message
    const messagesBefore = messages.slice(0, messageIndex);
    let userMsgIndex = -1;
    for (let i = messagesBefore.length - 1; i >= 0; i--) {
      if (messagesBefore[i].type === "human") {
        userMsgIndex = i;
        break;
      }
    }
    if (userMsgIndex === -1) return;

    const userMessage = messages[userMsgIndex].content;
    const sessionId = activeSessionId;
    if (!sessionId) return;

    // Remove messages after the user message (including the AI response)
    setMessages((prev) => prev.slice(0, userMsgIndex + 1));
    setIsLoading(true);

    try {
      const response = await sendMessage(sessionId, userMessage);
      const aiMsg: Message = { type: "ai", content: response.generation };
      setMessages((prev) => [...prev, aiMsg]);
      setLawDomain(response.law_domain);
      await loadSessions();
      showToast({ type: "success", title: "Regenerated response" });
    } catch (err) {
      console.error("Regenerate error:", err);
      showToast({ type: "error", title: "Failed to regenerate" });
    } finally {
      setIsLoading(false);
    }
  }, [messages, activeSessionId, loadSessions, showToast]);

  // Listen for regenerate events from ChatArea
  useEffect(() => {
    const handleRegenerateEvent = (event: Event) => {
      const customEvent = event as CustomEvent<{ index: number }>;
      handleRegenerate(customEvent.detail.index);
    };
    window.addEventListener("regenerate-message", handleRegenerateEvent as EventListener);
    return () => window.removeEventListener("regenerate-message", handleRegenerateEvent as EventListener);
  }, [handleRegenerate]);

  const handleSuggestionClick = useCallback((text: string) => {
    handleSendMessage(text);
  }, [handleSendMessage]);

  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  /* ── Render ── */
  return (
    <>
      {phase === "splash" && <SplashScreen onComplete={handleSplashComplete} />}

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
          <ChatInput
            onSend={handleSendMessage}
            disabled={isLoading}
            hasSession={!!activeSessionId}
          />
          <div ref={messagesEndRef} />
        </main>
      </div>
    </>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </ErrorBoundary>
  );
}