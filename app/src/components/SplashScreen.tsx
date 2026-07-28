import { useEffect, useState, useCallback } from "react";

interface SplashScreenProps {
  onComplete: () => void;
}

interface HealthStatus {
  status: "checking" | "healthy" | "unhealthy" | "error";
  message: string;
}

const HEALTH_CHECK_TIMEOUT = 5000;
const MIN_DISPLAY_TIME = 1500; // Minimum time to show splash for smooth UX

export default function SplashScreen({ onComplete }: SplashScreenProps) {
  const [health, setHealth] = useState<HealthStatus>({ status: "checking", message: "Checking backend..." });
  const [progress, setProgress] = useState(0);
  const [canSkip, setCanSkip] = useState(false);
  const [startTime] = useState(Date.now());

  const checkHealth = useCallback(async () => {
    try {
      setHealth({ status: "checking", message: "Connecting to backend..." });
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT);

      const res = await fetch(`${import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        await res.json();
        setHealth({ status: "healthy", message: "Backend ready" });
      } else {
        throw new Error(`Health check failed: ${res.status}`);
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setHealth({ status: "error", message: "Connection timeout" });
      } else {
        setHealth({ status: "error", message: "Backend unavailable" });
      }
    }
  }, []);

  const animateProgress = useCallback(() => {
    let progress = 0;
    const increment = 100 / (HEALTH_CHECK_TIMEOUT / 50);
    const interval = setInterval(() => {
      progress = Math.min(progress + increment, 95);
      setProgress(progress);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const cleanup = animateProgress();
    checkHealth();

    return () => {
      cleanup();
    };
  }, [checkHealth, animateProgress]);

  useEffect(() => {
    if (health.status === "healthy") {
      setProgress(100);
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, MIN_DISPLAY_TIME - elapsed);
      setTimeout(() => {
        onComplete();
      }, remaining);
    } else if (health.status === "error" || health.status === "unhealthy") {
      // Even on error, allow proceeding but warn user
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, MIN_DISPLAY_TIME - elapsed);
      setTimeout(() => {
        setCanSkip(true);
      }, remaining);
    }
  }, [health.status, onComplete, startTime]);

  const handleSkip = () => {
    if (canSkip) {
      onComplete();
    }
  };

  const getStatusColor = () => {
    switch (health.status) {
      case "healthy":
        return "var(--accent-gold)";
      case "error":
      case "unhealthy":
        return "var(--danger)";
      default:
        return "var(--accent-gold)";
    }
  };

  return (
    <div className="splash-screen" id="splash-screen">
      <div className="splash-content">
        <div className="splash-logo-wrapper">
          <span className="splash-logo">Ma'at</span>
          <div className="splash-ring" style={{ borderTopColor: getStatusColor() }} />
        </div>

        <span className="splash-subtitle">Indian Legal AI</span>

        <div className="splash-status" style={{ color: getStatusColor() }}>
          {health.message}
        </div>

        <div className="splash-progress-wrapper">
          <div className="splash-progress-track">
            <div
              className="splash-progress-bar"
              style={{
                width: `${progress}%`,
                background: health.status === "error" ? "var(--danger)" : "var(--accent-gold)",
              }}
            />
          </div>
        </div>

        {canSkip && (
          <button className="splash-skip-btn" onClick={handleSkip}>
            Continue anyway
          </button>
        )}

        {health.status === "error" && (
          <div className="splash-warning">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            <span>Backend may be unreachable. Some features might not work.</span>
          </div>
        )}
      </div>
    </div>
  );
}