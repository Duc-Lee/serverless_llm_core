from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()
request_buffer = asyncio.Queue()

async def check_vram_availability(model_id: str):
    await asyncio.sleep(0.001)
    return "192.168.1.100"

async def dispatch_to_node(node_ip: str, request_data: dict):
    await asyncio.sleep(0.05) 
    return {"token": "Hello"}

@app.post("/generate/{model_id}")
async def generate(model_id: str, prompt: str):
    # dua request vao queue thay vi drop neu vram day
    await request_buffer.put({"model_id": model_id, "prompt": prompt})
    current_req = await request_buffer.get()
    
    target_node = await check_vram_availability(model_id)
    if not target_node:
        raise HTTPException(status_code=503, detail="VRAM full")
        
    result = await dispatch_to_node(target_node, current_req)
    return {"result": result}
