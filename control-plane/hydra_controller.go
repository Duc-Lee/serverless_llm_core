package main

import (
	"fmt"
	"time"
)

type HydraModel struct {
	Name        string
	ModelPath   string
	TotalLayers int
}

type GlobalPageTable struct {
	NodeVRAM map[string]uint64
}

func main() {
	pageTable := &GlobalPageTable{NodeVRAM: make(map[string]uint64)}
	for {
		event := fetchNextHydraModelEvent()
		if event != nil {
			handleModelDeployment(event, pageTable)
		}
		time.Sleep(2 * time.Second)
	}
}

func handleModelDeployment(model *HydraModel, table *GlobalPageTable) {
	targetNodeIP := findNodeForWarmup(table)
	// bypass k8s pod lifecycle, goi thang xuong node de map vram
	sendWarmupRPCToWorker(targetNodeIP, model)
}

func findNodeForWarmup(table *GlobalPageTable) string {
	return "192.168.1.100" 
}

func sendWarmupRPCToWorker(nodeIP string, model *HydraModel) {
	fmt.Printf("[RPC -> %s] warmup %s\n", nodeIP, model.Name)
}

func fetchNextHydraModelEvent() *HydraModel {
	return &HydraModel{Name: "LLaMA-3-8B", ModelPath: "/mnt/nvme-of/models/llama3-8b.raw", TotalLayers: 32}
}
