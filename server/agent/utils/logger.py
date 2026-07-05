import os
import json
from datetime import datetime

# Calculate project root dynamically based on this file's location
# /server/agent/utils/logger.py -> ../../../logs
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

def log_node_event(node_name: str, status: str, tokens_used: int = None, error_payload: str = "None"):
    """
    Logs structured JSON operational metrics for a specific node event.
    Saved to: logs/{YYYY-MM-DD}_{node_name}.log
    """
    ensure_log_dir()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{date_str}_{node_name}.log")
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node_id": node_name,
        "status": status,
        "tokens_used": tokens_used,
        "error_payload": error_payload
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        # Fallback to standard print if file access fails to prevent crashing
        print(f"FAILED TO WRITE LOG: {e}")


def log_system_error(error_trace: str):
    """
    Logs full stack traces and system errors to a centralized system_errors.log.
    """
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, "system_errors.log")
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    try:
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] SYSTEM ERROR:\n{error_trace}\n{'-'*60}\n")
    except Exception as e:
        print(f"FAILED TO WRITE SYSTEM ERROR LOG: {e}")
