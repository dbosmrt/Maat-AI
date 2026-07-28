import { Message } from "../api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRef, useEffect, useCallback, useState, useMemo } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  onSuggestionClick: (text: string) => void;
  lawDomain: string | null;
}

const SUGGESTIONS = [
  { icon: "⚖️", text: "What are the grounds for divorce under Hindu Marriage Act?" },
  { icon: "📜", text: "Explain the difference between IPC Section 302 and 304" },
  { icon: "🏠", text: "What are a tenant's rights under the Rent Control Act?" },
  { icon: "💼", text: "What legal remedies exist for wrongful termination?" },
];

export default function ChatArea({ messages, isLoading, onSuggestionClick, lawDomain }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const hasMessages = messages.length > 0;

  const markdownComponents = useMemo(() => ({
    code: ({ node, inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        <SyntaxHighlighter
          language={match[1]}
          style={atomDark}
          customStyle={{
            margin: "0.5em 0",
            borderRadius: "var(--radius-md)",
            fontSize: "0.8rem",
            lineHeight: "1.6",
          }}
          showLineNumbers={false}
          wrapLongLines={true}
          {...props}
        >
          {String(node.value).replace(/\n$/, "")}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
    pre: ({ children, ...props }: any) => (
      <div className="markdown-pre" {...props}>
        {children}
      </div>
    ),
    blockquote: ({ children, ...props }: any) => (
      <blockquote className="markdown-blockquote" {...props}>
        {children}
      </blockquote>
    ),
    hr: () => <hr className="markdown-hr" />,
    table: ({ children, ...props }: any) => (
      <div className="markdown-table-wrapper">
        <table {...props}>{children}</table>
      </div>
    ),
  }), []);

  return (
    <div className="chat-area" id="chat-area" ref={messagesContainerRef}>
      {/* Header */}
      <div className="chat-header">
        <span className="chat-header-title">
          {hasMessages ? "Chat with Ma'at" : "New Conversation"}
        </span>
        {lawDomain && lawDomain !== "General" && (
          <span className="chat-header-badge">{lawDomain}</span>
        )}
      </div>

      {/* Messages or Welcome */}
      {hasMessages ? (
        <div className="messages-container" id="messages-container" role="log" aria-live="polite" aria-label="Conversation">
          {messages.map((msg, idx) => (
            <MessageBubble
              key={`${msg.type}-${idx}-${msg.content.slice(0, 20)}`}
              message={msg}
              index={idx}
              markdownComponents={markdownComponents}
            />
          ))}

          {/* Typing indicator */}
          {isLoading && (
            <div className="typing-indicator" aria-live="polite" aria-label="AI is typing">
              <div className="message-avatar ai" aria-hidden="true">M</div>
              <div className="typing-dots">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      ) : (
        <div className="welcome-container" id="welcome-container">
          <div className="welcome-icon" aria-hidden="true">M</div>
          <h1 className="welcome-title">Welcome to Ma'at</h1>
          <p className="welcome-subtitle">
            Your AI-powered Indian Legal Assistant. Ask me anything about Indian law —
            from constitutional rights to criminal procedures.
          </p>
          <div className="welcome-suggestions">
            {SUGGESTIONS.map((s, idx) => (
              <button
                key={idx}
                className="suggestion-card"
                onClick={() => onSuggestionClick(s.text)}
                id={`suggestion-${idx}`}
              >
                <div className="suggestion-icon" aria-hidden="true">{s.icon}</div>
                <div className="suggestion-text">{s.text}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MessageBubbleProps {
  message: Message;
  index: number;
  markdownComponents: Record<string, any>;
}

function MessageBubble({ message, index, markdownComponents }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }, [message.content]);

  const handleRegenerate = useCallback(() => {
    window.dispatchEvent(new CustomEvent("regenerate-message", { detail: { index } }));
  }, [index]);

  return (
    <div key={index} className={`message-row ${message.type}`}>
      {message.type === "ai" && (
        <div className="message-avatar ai" aria-hidden="true">M</div>
      )}
      <div className="message-bubble-wrapper">
        <div className={`message-bubble ${message.type} markdown-body`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {message.content}
          </ReactMarkdown>
        </div>
        <div className="message-actions">
          <button
            className={`message-action-btn ${copied ? "copied" : ""}`}
            onClick={handleCopy}
            aria-label={copied ? "Copied!" : "Copy message"}
            title={copied ? "Copied!" : "Copy"}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
          </button>
          {message.type === "ai" && (
            <button
              className="message-action-btn"
              onClick={handleRegenerate}
              aria-label="Regenerate response"
              title="Regenerate"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14">
                <path d="M23 4v6h-6" />
                <path d="M1 20v-6h6" />
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
              </svg>
            </button>
          )}
        </div>
      </div>
      {message.type === "human" && (
        <div className="message-avatar human" aria-hidden="true">You</div>
      )}
    </div>
  );
}