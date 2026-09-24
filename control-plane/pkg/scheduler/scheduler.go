package scheduler

import (
	"errors"
	"log"
)

type Scheduler struct {
	workerAddr string
}

func NewScheduler(workerAddr string) *Scheduler {
	return &Scheduler{workerAddr: workerAddr}
}

func (s *Scheduler) RouteRequest(modelID string) (string, error) {
	log.Printf("[Scheduler] Routing request for %s to Worker Node: %s", modelID, s.workerAddr)
	if s.workerAddr == "" {
		return "", errors.New("no available workers")
	}
	return s.workerAddr, nil
}
