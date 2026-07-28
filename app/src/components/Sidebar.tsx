import { useState, useRef, useEffect, ChangeEvent, KeyboardEvent, MouseEvent } from "react";
import { SessionItem } from "../api";

interface SidebarProps {
  sessions: SessionItem[];
  activeSessionId: string | null;
  sidebarOpen: boolean;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onToggleSidebar: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  sidebarOpen,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onToggleSidebar,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);
  const sessionListRef = useRef<HTMLDivElement>(null);
  const focusedIndexRef = useRef(-1);

  const filteredSessions = sessions.filter((session) =>
    session.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Focus search input when sidebar opens
  useEffect(() => {
    if (sidebarOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [sidebarOpen]);

  // Keyboard navigation for session list
  useEffect(() => {
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (!sidebarOpen) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const items = filteredSessions;
      if (items.length === 0) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          focusedIndexRef.current = Math.min(focusedIndexRef.current + 1, items.length - 1);
          break;
        case "ArrowUp":
          e.preventDefault();
          focusedIndexRef.current = Math.max(focusedIndexRef.current - 1, 0);
          break;
        case "Enter":
          e.preventDefault();
          if (focusedIndexRef.current >= 0 && focusedIndexRef.current < items.length) {
            onSelectSession(items[focusedIndexRef.current].session_id);
          }
          break;
        case "Escape":
          e.preventDefault();
          onToggleSidebar();
          break;
        case "n":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            onNewChat();
          }
          break;
        case "/":
          if (document.activeElement !== searchInputRef.current) {
            e.preventDefault();
            searchInputRef.current?.focus();
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sidebarOpen, filteredSessions, onNewChat, onSelectSession, onToggleSidebar]);

  // Scroll focused item into view
  useEffect(() => {
    if (focusedIndexRef.current >= 0 && sessionListRef.current) {
      const items = sessionListRef.current.querySelectorAll(".session-item");
      const focusedItem = items[focusedIndexRef.current] as HTMLElement;
      if (focusedItem) {
        focusedItem.scrollIntoView({ block: "nearest" });
      }
    }
  }, [focusedIndexRef.current, filteredSessions]);

  const handleSearchChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    focusedIndexRef.current = -1;
  };

  const handleRenameStart = (e: MouseEvent, session: SessionItem) => {
    e.stopPropagation();
    setRenamingSessionId(session.session_id);
    setRenameValue(session.preview);
  };

  const handleRenameConfirm = (_sessionId: string) => {
    if (renameValue.trim() && renameValue !== sessions.find(s => s.session_id === renamingSessionId)?.preview) {
      // TODO: Implement rename API call
      // For now, just update locally
      console.log("Rename session:", renamingSessionId, "to:", renameValue);
    }
    setRenamingSessionId(null);
  };

  const handleRenameCancel = () => {
    setRenamingSessionId(null);
  };

  const handleRenameKeyDown = (e: KeyboardEvent<HTMLInputElement>, sessionId: string) => {
    if (e.key === "Enter") {
      handleRenameConfirm(sessionId);
    } else if (e.key === "Escape") {
      handleRenameCancel();
    }
  };

  const handleDeleteClick = (e: MouseEvent, sessionId: string) => {
    e.stopPropagation();
    setShowDeleteConfirm(sessionId);
  };

  const handleDeleteConfirm = (sessionId: string) => {
    onDeleteSession(sessionId);
    setShowDeleteConfirm(null);
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(null);
  };

  const formatDate = () => {
    // Extract timestamp from session ID if possible, or use current time
    // For now, return a formatted date
    return new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  return (
    <>
      {/* Mobile hamburger toggle */}
      <button
        className="sidebar-toggle"
        onClick={onToggleSidebar}
        id="sidebar-toggle"
        aria-label="Toggle sidebar"
        aria-expanded={sidebarOpen}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <aside
        className={`sidebar ${sidebarOpen ? "open" : ""}`}
        id="sidebar"
        role="complementary"
        aria-label="Chat sessions"
      >
        <div className="sidebar-header">
          {/* Brand */}
          <div className="sidebar-brand">
            <div className="sidebar-brand-icon" aria-hidden="true">M</div>
            <span className="sidebar-brand-text">Ma'at</span>
          </div>

          {/* New Chat button */}
          <button
            className="new-chat-btn"
            onClick={onNewChat}
            id="new-chat-btn"
            aria-label="Start new chat (Ctrl+N)"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>New Chat</span>
            <kbd className="shortcut-hint">Ctrl+N</kbd>
          </button>
        </div>

        {/* Search */}
        <div className="sidebar-search">
          <label htmlFor="session-search" className="visually-hidden">Search sessions</label>
          <div className="search-input-wrapper">
            <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              ref={searchInputRef}
              type="search"
              id="session-search"
              className="session-search-input"
              placeholder="Search sessions... (Ctrl+/)"
              value={searchQuery}
              onChange={handleSearchChange}
              aria-label="Search sessions"
            />
            {searchQuery && (
              <button
                className="search-clear-btn"
                onClick={() => setSearchQuery("")}
                aria-label="Clear search"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Session list */}
        <div className="session-list" id="session-list" ref={sessionListRef} role="listbox" aria-label="Chat sessions">
          {filteredSessions.length > 0 ? (
            <>
              <div className="session-list-label">Recent Chats</div>
              {filteredSessions.map((session, idx) => {
                const isActive = activeSessionId === session.session_id;
                const isFocused = focusedIndexRef.current === idx;
                const isRenaming = renamingSessionId === session.session_id;

                return (
                  <div
                    key={session.session_id}
                    className={`session-item ${isActive ? "active" : ""} ${isFocused ? "focused" : ""} ${isRenaming ? "renaming" : ""}`}
                    onClick={() => !isRenaming && onSelectSession(session.session_id)}
                    onDoubleClick={(e) => { e.stopPropagation(); handleRenameStart(e, session); }}
                    id={`session-${session.session_id}`}
                    role="option"
                    aria-selected={isActive}
                    aria-current={isActive ? "true" : undefined}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        if (!isRenaming) onSelectSession(session.session_id);
                      }
                    }}
                  >
                    <div className="session-item-content">
                      {isRenaming ? (
                        <input
                          type="text"
                          className="session-rename-input"
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => handleRenameKeyDown(e, session.session_id)}
                          onBlur={() => handleRenameConfirm(session.session_id)}
                          autoFocus
                          aria-label="Rename session"
                        />
                      ) : (
                        <>
                          <div className="session-item-preview">{session.preview}</div>
                          <div className="session-item-meta">
                            <span>{session.message_count} message{session.message_count !== 1 ? "s" : ""}</span>
                            <span className="session-date">· {formatDate()}</span>
                          </div>
                        </>
                      )}
                    </div>
                    {!isRenaming && (
                      <div className="session-item-actions">
                        <button
                          className="session-action-btn rename-btn"
                          onClick={(e) => handleRenameStart(e, session)}
                          aria-label={`Rename "${session.preview}"`}
                          title="Rename"
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                        </button>
                        <button
                          className="session-action-btn delete-btn"
                          onClick={(e) => handleDeleteClick(e, session.session_id)}
                          aria-label={`Delete "${session.preview}"`}
                          title="Delete"
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                          </svg>
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </>
          ) : (
            <div className="sessions-empty">
              <div className="sessions-empty-icon" aria-hidden="true">💬</div>
              <div className="sessions-empty-text">
                {searchQuery ? "No matching conversations." : "No conversations yet."}
                <br />
                <button className="empty-state-btn" onClick={onNewChat}>
                  {searchQuery ? "Clear search" : "Start a new chat"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Delete confirmation modal */}
        {showDeleteConfirm && (
          <div className="modal-overlay" onClick={handleDeleteCancel} role="dialog" aria-modal="true" aria-labelledby="delete-modal-title">
            <div className="modal confirm-modal" onClick={(e) => e.stopPropagation()}>
              <h3 id="delete-modal-title" className="modal-title">Delete Conversation?</h3>
              <p className="modal-text">
                This will permanently delete the conversation. This action cannot be undone.
              </p>
              <div className="modal-actions">
                <button className="modal-btn cancel" onClick={handleDeleteCancel}>
                  Cancel
                </button>
                <button className="modal-btn danger" onClick={() => handleDeleteConfirm(showDeleteConfirm)}>
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Rename confirmation toast would go here */}

        <div className="sidebar-footer">
          <div className="footer-links">
            <a href="#" className="footer-link">Keyboard Shortcuts</a>
            <span className="footer-separator" aria-hidden="true">·</span>
            <a href="#" className="footer-link">Privacy</a>
            <span className="footer-separator" aria-hidden="true">·</span>
            <a href="#" className="footer-link">Terms</a>
          </div>
          <div className="sidebar-version">Ma'at Legal AI · v1.0</div>
        </div>
      </aside>

      {/* Keyboard shortcuts help overlay */}
      <ShortcutsHelp isOpen={sidebarOpen} />
    </>
  );
}

function ShortcutsHelp({ isOpen }: { isOpen: boolean }) {
  const shortcuts = [
    { keys: ["Ctrl", "N"], description: "New Chat" },
    { keys: ["Ctrl", "/"], description: "Focus Search" },
    { keys: ["↑", "↓"], description: "Navigate Sessions" },
    { keys: ["Enter"], description: "Open Session" },
    { keys: ["Esc"], description: "Close Sidebar" },
    { keys: ["Ctrl", "Enter"], description: "Send Message" },
    { keys: ["Shift", "Enter"], description: "New Line" },
  ];

  return (
    <div className={`shortcuts-help ${isOpen ? "visible" : ""}`} aria-hidden="true">
      <div className="shortcuts-content">
        <h4>Keyboard Shortcuts</h4>
        <ul>
          {shortcuts.map((s, i) => (
            <li key={i}>
              <kbd className="shortcut-key">{s.keys.join(" + ")}</kbd>
              <span>{s.description}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}