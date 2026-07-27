import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  hasSession: boolean;
}

export default function ChatInput({ onSend, disabled, hasSession }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    }
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
  };

  const isDisabled = disabled || !hasSession;
  const placeholder = hasSession
    ? "Ask Ma'at a legal question..."
    : "Create a new chat session first using the 'New Chat' button in the sidebar";

  return (
    <div className="chat-input-wrapper">
      <div className="chat-input-container" id="chat-input-container">
        <textarea
          ref={textareaRef}
          className="chat-input-textarea"
          id="chat-input"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={isDisabled}
        />
        <button
          className="send-btn"
          id="send-btn"
          onClick={handleSend}
          disabled={isDisabled || !value.trim()}
          aria-label="Send message"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
      <div className="chat-input-hint">
        Ma'at provides AI-generated legal information. Always consult a qualified lawyer for legal advice.
      </div>
    </div>
  );
}
