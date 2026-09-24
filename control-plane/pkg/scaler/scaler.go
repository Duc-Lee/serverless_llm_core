package scaler

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
)

type HydraScaler struct {
	engineCmd *exec.Cmd
	stdinPipe io.WriteCloser
}

func NewHydraScaler() *HydraScaler {
	return &HydraScaler{}
}

func (s *HydraScaler) Start() {
	s.engineCmd = exec.Command("../build/Release/hydra_daemon.exe") 
	
	s.engineCmd.Stdout = os.Stdout
	s.engineCmd.Stderr = os.Stderr
	
	pipe, err := s.engineCmd.StdinPipe()
	if err != nil {
		log.Fatalf("Failed to create STDIN pipe: %v", err)
	}
	s.stdinPipe = pipe
	
	err = s.engineCmd.Start()
	if err != nil {
		log.Printf("Failed to start daemon: %v. Running in mock mode.", err)
	} else {
		log.Println("Scaler: Daemon started, IPC pipe connected.")
	}
}

func (s *HydraScaler) ForwardToEngine(model string, prompt string) error {
	modelHash := uint32(2166136261) 
	
	estimatedTokens := len(prompt) / 4
	if estimatedTokens == 0 {
		estimatedTokens = 64
	}
	
	if s.stdinPipe != nil {
		command := fmt.Sprintf("%d %d\n", modelHash, estimatedTokens)
		_, err := s.stdinPipe.Write([]byte(command))
		if err != nil {
			return err
		}
	}
	
	return nil
}
