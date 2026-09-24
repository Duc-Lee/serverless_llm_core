import socket
import time
import threading
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

class ClusterStateManager:
    def __init__(self, idle_timeout=10):
        self.state = "ZERO"  # "ZERO" | "ACTIVE"
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()
        self.lock = threading.Lock()
        
        # Chạy thread theo dõi scale down
        threading.Thread(target=self._idle_reaper_loop, daemon=True).start()

    def send_worker_cmd(self, cmd: str) -> str:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect("/tmp/hydra_worker.sock")
        s.sendall(cmd.encode())
        res = s.recv(1024).decode()
        s.close()
        return res

    def ensure_active(self):
        with self.lock:
            self.last_activity = time.time()
            if self.state == "ZERO":
                print("[Control-Plane] Cold-start triggered -> Issuing HYDRATE...")
                self.send_worker_cmd("HYDRATE:1")
                self.state = "ACTIVE"

    def _idle_reaper_loop(self):
        while True:
            time.sleep(1)
            with self.lock:
                if self.state == "ACTIVE" and (time.time() - self.last_activity > self.idle_timeout):
                    print("[Control-Plane] Idle timeout exceeded -> Issuing EVICT (Scale-to-Zero)...")
                    self.send_worker_cmd("EVICT")
                    self.state = "ZERO"

cluster = ClusterStateManager(idle_timeout=5)

@app.post("/v1/chat/completions")
async def chat_completion(payload: dict):
    # Đảm bảo GPU đã sẵn sàng trước khi nạp prompt
    cluster.ensure_active()
    
    # Mô phỏng quá trình forward inference
    return {
        "status": "success",
        "model": payload.get("model", "default"),
        "response": "Tokens generated successfully via Hydra Worker."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
