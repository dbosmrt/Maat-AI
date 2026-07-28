import { useState, useRef, useEffect, useCallback, KeyboardEvent, ChangeEvent, FocusEvent } from "react";
import { useKeyboardShortcuts, KeyboardShortcut } from "../hooks/useKeyboardShortcuts";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  hasSession: boolean;
}

const MAX_MESSAGE_LENGTH = 4000;

export default function ChatInput({ onSend, disabled, hasSession }: ChatInputProps) {
  const [value, setValue] = useState("");
  const [height, setHeight] = useState(24);
  const [isFocused, setIsFocused] = useState(false);
  const [showCharCount, setShowCharCount] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputContainerRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const newHeight = Math.min(textarea.scrollHeight, 120);
      setHeight(newHeight);
      textarea.style.height = `${newHeight}px`;
    }
  }, [value]);

  // Auto-focus on mount and when session becomes available
  useEffect(() => {
    if (hasSession && textareaRef.current && !disabled) {
      textareaRef.current.focus();
    }
  }, [hasSession, disabled]);

  // Handle keyboard shortcuts
  const shortcuts: KeyboardShortcut[] = [
    {
      key: "Enter",
      ctrl: true,
      action: () => handleSend(),
      description: "Send message",
    },
    {
      key: "Enter",
      meta: true,
      action: () => handleSend(),
      description: "Send message (Mac)",
    },
    {
      key: "/",
      ctrl: true,
      action: () => textareaRef.current?.focus(),
      description: "Focus input",
    },
    {
      key: "k",
      ctrl: true,
      action: () => textareaRef.current?.focus(),
      description: "Focus input (alternative)",
    },
  ];

  useKeyboardShortcuts(shortcuts, !disabled);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    setHeight(24);
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
  }, []);

  const handleFocus = useCallback((_e: FocusEvent<HTMLTextAreaElement>) => {
    setIsFocused(true);
    if (value.length > 100) setShowCharCount(true);
  }, [value.length]);

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    setShowCharCount(false);
  }, []);

  const charCount = value.length;
  const isNearLimit = charCount > MAX_MESSAGE_LENGTH * 0.8;
  const isOverLimit = charCount > MAX_MESSAGE_LENGTH;
  const isDisabled = disabled || !hasSession || charCount === 0 || isOverLimit;

  const placeholder = hasSession
    ? "Ask Ma'at a legal question... (Ctrl+Enter to send)"
    : "Create a new chat session first using the 'New Chat' button in the sidebar";

  return (
    <div className="chat-input-wrapper">
      <div
        className={`chat-input-container ${isFocused ? "focused" : ""} ${isNearLimit ? "near-limit" : ""} ${
          isOverLimit ? "over-limit" : ""
        }`}
        ref={inputContainerRef}
        id="chat-input-container"
      >
        <textarea
          ref={textareaRef}
          className="chat-input-textarea"
          id="chat-input"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          rows={1}
          disabled={disabled || !hasSession}
          maxLength={MAX_MESSAGE_LENGTH}
          style={{ height: `${height}px` }}
          aria-label="Message input"
          aria-describedby={showCharCount ? "char-count" : undefined}
        />
        <button
          className="send-btn"
          id="send-btn"
          onClick={handleSend}
          disabled={isDisabled}
          aria-label="Send message"
          title="Send (Ctrl+Enter)"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>

      <div className="chat-input-footer">
        {showCharCount && (
          <div
            id="char-count"
            className={`char-count ${isNearLimit ? "warning" : ""} ${isOverLimit ? "error" : ""}`}
            aria-live="polite"
          >
            {charCount} / {MAX_MESSAGE_LENGTH}
          </div>
        )}

        <div className="chat-input-hints">
          <kbd className="hint-key">
            <span className="key-mod">Ctrl</span> + <span className="key-main">Enter</span>
          </kbd>
          <span className="hint-desc">Send</span>
          <span className="hint-sep">|</span>
          <kbd className="hint-key">
            <span className="key-mod">Shift</span> + <span className="key-main">Enter</span>
          </kbd>
          <span className="hint-desc">New line</span>
          <span className="hint-sep">|</span>
          <kbd className="hint-key">
            <span className="key-mod">Ctrl</span> + <span className="key-main">/</span>
          </kbd>
          <span className="hint-desc">Focus input</span>
        </div>

        <div className="chat-input-disclaimer">
          Ma'at provides AI-generated legal information. Always consult a qualified lawyer for legal advice.
        </div>
      </div>
    </div>
  );
}