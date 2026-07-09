import os
import sys
import uuid
import json
import logging
from agent.utils.logger import get_logger
from typing import List

# Suppress verbose node logging to keep the terminal UI clean
logging.getLogger("agent.node").setLevel(logging.WARNING)
logging.getLogger("agent.chat_graph").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent.chat_graph import build_chat_graph
from agent.state import AgentState

CHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chats")
os.makedirs(CHAT_DIR, exist_ok=True)

class TerminalChat:
    def __init__(self):
        self.app = build_chat_graph()
        self.session_id = None
        self.history: List[BaseMessage] = []
        self.memory_summary = ""

    def start_new_session(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.history = []
        self.memory_summary = ""
        print(f"\n[+] Started new chat session: {self.session_id}")

    def list_sessions(self):
        files = [f for f in os.listdir(CHAT_DIR) if f.endswith('.json')]
        if not files:
            print("[-] No previous sessions found.")
            return
            
        print("\n=== Previous Sessions ===")
        for f in files:
            print(f" - {f.replace('.json', '')}")
        print("=========================\n")

    def load_session(self, sid: str):
        filepath = os.path.join(CHAT_DIR, f"{sid}.json")
        if not os.path.exists(filepath):
            print(f"[-] Session {sid} not found.")
            return
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.session_id = sid
        self.memory_summary = data.get("memory_summary", "")
        
        # Load raw history
        self.history = []
        for msg in data.get("history", []):
            if msg["type"] == "human":
                self.history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                self.history.append(AIMessage(content=msg["content"]))
                
        print(f"\n[+] Loaded session {sid} (History: {len(self.history)} messages)")

    def save_session(self):
        if not self.session_id:
            return
            
        filepath = os.path.join(CHAT_DIR, f"{self.session_id}.json")
        
        # Serialize history
        serializable_history = []
        for msg in self.history:
            msg_type = "human" if isinstance(msg, HumanMessage) else "ai"
            serializable_history.append({"type": msg_type, "content": msg.content})
            
        data = {
            "session_id": self.session_id,
            "memory_summary": self.memory_summary,
            "history": serializable_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def chat_loop(self):
        print("=====================================================")
        print("   Ma-at: Indian Legal AI Assistant (Terminal UI)    ")
        print("=====================================================")
        print("Commands:")
        print("  /new         - Start a new chat")
        print("  /list        - List previous chats")
        print("  /load <id>   - Load a specific chat")
        print("  /exit        - Close the application")
        print("-----------------------------------------------------\n")
        
        self.start_new_session()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == "/exit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "/new":
                    self.start_new_session()
                    continue
                elif user_input.lower() == "/list":
                    self.list_sessions()
                    continue
                elif user_input.lower().startswith("/load"):
                    parts = user_input.split(" ")
                    if len(parts) == 2:
                        self.load_session(parts[1])
                    else:
                        print("[-] Usage: /load <session_id>")
                    continue
                
                # Append user message to history
                self.history.append(HumanMessage(content=user_input))
                
                # Prepare state for the graph
                state = AgentState(
                    query=user_input,
                    session_id=self.session_id,
                    chat_history=self.history[-4:], # Only pass the last 4 messages to prevent context overflow
                    memory_summary=self.memory_summary,
                    is_scenario=False,
                    requires_case_law=False,
                    search_required=False,
                    documents=[],
                    case_laws=[],
                    generation="",
                    iteration_count=0
                )
                
                print("\nMa-at is thinking...", end="", flush=True)
                
                # Execute graph
                result = self.app.invoke(state)
                
                # Clear thinking line
                print("\r" + " " * 30 + "\r", end="") 
                
                # Print generation
                generation = result.get("generation", "Error generating response.")
                print(f"Ma-at: {generation}\n")
                
                # Append AI response to history
                self.history.append(AIMessage(content=generation))
                
                # Note: In a real system we would update self.memory_summary here using an LLM 
                # if self.history gets too long.
                
                # Save session state
                self.save_session()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\n[-] An error occurred: {e}")

if __name__ == "__main__":
    ui = TerminalChat()
    ui.chat_loop()
