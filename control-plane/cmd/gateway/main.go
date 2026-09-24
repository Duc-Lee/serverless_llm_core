package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"hydra/control-plane/pkg/scaler"
)

type CompletionRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

func main() {
	hydraScaler := scaler.NewHydraScaler()
	hydraScaler.Start()

	http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		start := time.Now()

		var req CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		log.Printf("Gateway: Request received for model %s", req.Model)

		err := hydraScaler.ForwardToEngine(req.Model, req.Prompt)
		if err != nil {
			http.Error(w, "Inference Engine failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model": "%s", "response": "Generated output", "ttft_ms": %v}`,
			req.Model, time.Since(start).Milliseconds())
	})
	log.Println("Gateway: Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
