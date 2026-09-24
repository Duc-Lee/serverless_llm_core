package main

import (
	"log"
	"net"
	"net/http"
	"time"

	"hydra/control-plane/pkg/autoscaler"
	"hydra/control-plane/pkg/scheduler"
)

type UnixSocketClient struct {
	socketPath string
}

func (u *UnixSocketClient) sendCommand(cmd string) error {
	conn, err := net.Dial("unix", u.socketPath)
	if err != nil {
		log.Printf("[IPC Error] Could not connect to worker via UDS: %v", err)
		return err
	}
	defer conn.Close()
	_, err = conn.Write([]byte(cmd))
	return err
}

func (u *UnixSocketClient) Warmup(modelID string) error {
	log.Printf("[IPC Client] Request Queued. Sending ATTACH_WEIGHTS/REMAP for %s...", modelID)
	return u.sendCommand("REMAP " + modelID)
}

func (u *UnixSocketClient) Evict(modelID string) error {
	log.Printf("[IPC Client] Idle timeout reached. Sending UNMAP (Scale-to-Zero) for %s...", modelID)
	return u.sendCommand("UNMAP " + modelID)
}

func main() {
	sched := scheduler.NewScheduler("/tmp/hydra.sock")
	client := &UnixSocketClient{socketPath: "/tmp/hydra.sock"}
	
	// Scale-to-zero after 30 seconds of inactivity
	asc := autoscaler.NewAutoscaler(client, 30*time.Second)

	http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		modelID := r.URL.Query().Get("model")
		if modelID == "" {
			modelID = "llama-3-8b"
		}
		
		asc.RecordActivity(modelID)
		
		workerPath, err := sched.RouteRequest(modelID)
		if err != nil {
			http.Error(w, "No worker available", http.StatusServiceUnavailable)
			return
		}
		
		// Send ATTACH_WEIGHTS to map memory before forwarding request
		client.Warmup(modelID)
		
		w.Write([]byte("Routed to worker at: " + workerPath + "\n"))
	})

	log.Println("[Gateway] Routing & Request Queuing Gateway listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
