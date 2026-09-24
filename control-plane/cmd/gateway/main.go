package main

import (
	"log"
	"net/http"
	"time"

	"hydra/control-plane/pkg/autoscaler"
	"hydra/control-plane/pkg/scheduler"
)

type MockWorkerClient struct{}

func (m *MockWorkerClient) Warmup(modelID string) error {
	log.Printf("[IPC Client] Sending WARMUP signal for %s to C++ Worker (Map VRAM)...", modelID)
	return nil
}

func (m *MockWorkerClient) Evict(modelID string) error {
	log.Printf("[IPC Client] Sending EVICT signal for %s to C++ Worker (Unmap VRAM)...", modelID)
	return nil
}

func main() {
	sched := scheduler.NewScheduler("localhost:9000")
	client := &MockWorkerClient{}
	
	// Scale-to-zero after 30 seconds of inactivity
	asc := autoscaler.NewAutoscaler(client, 30*time.Second)

	http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		modelID := r.URL.Query().Get("model")
		if modelID == "" {
			modelID = "llama-3-8b"
		}
		
		asc.RecordActivity(modelID)
		
		worker, err := sched.RouteRequest(modelID)
		if err != nil {
			http.Error(w, "No worker available", http.StatusServiceUnavailable)
			return
		}
		
		client.Warmup(modelID)
		w.Write([]byte("Routed to worker: " + worker + "\n"))
	})

	log.Println("[Gateway] Ingress Control Plane & Request Queue listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
