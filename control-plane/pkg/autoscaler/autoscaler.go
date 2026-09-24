package autoscaler

import (
	"log"
	"sync"
	"time"
)

type WorkerClient interface {
	Warmup(modelID string) error
	Evict(modelID string) error
}

type Autoscaler struct {
	client       WorkerClient
	idleTimeout  time.Duration
	lastActivity map[string]time.Time
	mu           sync.Mutex
}

func NewAutoscaler(client WorkerClient, timeout time.Duration) *Autoscaler {
	a := &Autoscaler{
		client:       client,
		idleTimeout:  timeout,
		lastActivity: make(map[string]time.Time),
	}
	go a.monitorLoop()
	return a
}

func (a *Autoscaler) RecordActivity(modelID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.lastActivity[modelID] = time.Now()
}

func (a *Autoscaler) monitorLoop() {
	ticker := time.NewTicker(5 * time.Second)
	for range ticker.C {
		a.mu.Lock()
		now := time.Now()
		for modelID, lastSeen := range a.lastActivity {
			if now.Sub(lastSeen) > a.idleTimeout {
				log.Printf("[Autoscaler] Model %s idle for %v. Evicting (Scale-to-Zero)...", modelID, a.idleTimeout)
				a.client.Evict(modelID)
				delete(a.lastActivity, modelID)
			}
		}
		a.mu.Unlock()
	}
}
