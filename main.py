import os
import asyncio
import time
from io import BytesIO

import mujoco
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse


class MuJoCoSimulator:
    def __init__(self, model_path: str = None, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        
        if model_path and os.path.exists(model_path):
            # Load from file if path provided and exists
            print(f"Loading model from: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            # Use built-in simple scene
            print("Using built-in falling objects scene")
            self.model = mujoco.MjModel.from_xml_string("""
                <mujoco>
                    <option gravity="0 0 -9.81" timestep="0.01"/>
                    <visual>
                        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8"/>
                    </visual>
                    <worldbody>
                        <light pos="0 0 3" dir="0 0 -1" directional="true"/>
                        <geom name="floor" size="2 2 0.05" type="plane" material="grid"/>
                        <body name="box" pos="0 0 1.0">
                            <geom name="box_geom" size="0.1 0.1 0.1" type="box" rgba="1 0 0 1" mass="1"/>
                            <joint name="box_joint" type="free"/>
                        </body>
                        <body name="sphere" pos="0.5 0 2.0">
                            <geom name="sphere_geom" size="0.08" type="sphere" rgba="0 1 0 1" mass="0.5"/>
                            <joint name="sphere_joint" type="free"/>
                        </body>
                    </worldbody>
                    <asset>
                        <material name="grid" texture="grid"/>
                        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.8 0.8 0.8" rgb2="0.2 0.3 0.4"/>
                    </asset>
                </mujoco>
            """)
        
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # Set camera position
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.0
        self.camera.elevation = -20
        self.camera.azimuth = 45
        
    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    def render_frame(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)


# Global simulator - can be configured via environment variable
model_path = os.getenv('MUJOCO_MODEL_PATH')  # Set via: export MUJOCO_MODEL_PATH=/path/to/model.xml
simulator = MuJoCoSimulator(model_path=model_path)
app = FastAPI(title="MuJoCo Streaming Server")


@app.get("/frame")
async def get_frame():
    """Get a single frame as JPEG - optimized for speed."""
    frame_start = time.time()
    
    simulator.step()
    rgb_array = simulator.render_frame()
    
    # Faster JPEG encoding - lower quality but much faster
    image = Image.fromarray(rgb_array, mode='RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=50, optimize=True)  # Lower quality, no optimization
    
    frame_time = (time.time() - frame_start) * 1000  # Convert to ms
    
    return StreamingResponse(
        BytesIO(buffer.getvalue()),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Frame-Time": str(int(frame_time))  # Debug header
        }
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple HTML page that fetches frames using JavaScript."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuJoCo Live Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
                text-align: center;
            }
            .container {
                border: 2px solid #333;
                border-radius: 8px;
                display: inline-block;
                background-color: #000;
            }
            #stream {
                display: block;
                width: 640px;
                height: 480px;
            }
            button {
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
                border: none;
                border-radius: 4px;
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }
            button:hover { background-color: #0056b3; }
            .status {
                margin: 10px;
                padding: 10px;
                background-color: #fff;
                border-radius: 4px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <h1>MuJoCo Live Simulation</h1>
        
        <div class="container">
            <img id="stream" alt="MuJoCo Stream">
        </div>
        
        <div>
            <button onclick="toggleStream()">Start/Stop</button>
            <button onclick="resetSim()">Reset</button>
        </div>
        
        <div class="status" id="status">Ready to start...</div>
        
        <script>
            let streaming = false;
            let intervalId = null;
            const img = document.getElementById('stream');
            const status = document.getElementById('status');
            
            function updateStatus(msg) {
                status.textContent = new Date().toLocaleTimeString() + ': ' + msg;
            }
            
            let frameCount = 0;
            let startTime = Date.now();
            let totalFrameTime = 0;
            
            function fetchFrame() {
                const requestStart = Date.now();
                const timestamp = Date.now();
                img.src = '/frame?' + timestamp;
                
                img.onload = function() {
                    const requestTime = Date.now() - requestStart;
                    totalFrameTime += requestTime;
                    frameCount++;
                    
                    if (frameCount % 10 === 0) { // Update every 10 frames
                        const elapsed = Date.now() - startTime;
                        const actualFPS = (frameCount / elapsed * 1000).toFixed(1);
                        const avgFrameTime = (totalFrameTime / frameCount).toFixed(0);
                        updateStatus(`FPS: ${actualFPS}, Avg frame time: ${avgFrameTime}ms`);
                    }
                    
                    // More aggressive timing - target 30 FPS
                    if (streaming) {
                        const targetInterval = 33; // 30 FPS
                        const nextDelay = Math.max(1, targetInterval - requestTime); // Minimum 1ms delay
                        setTimeout(fetchFrame, nextDelay);
                    }
                };
                
                img.onerror = function() {
                    if (streaming) {
                        setTimeout(fetchFrame, 100); // Retry after error
                    }
                };
            }
            
            function toggleStream() {
                if (streaming) {
                    streaming = false;
                    updateStatus('Stream stopped');
                } else {
                    streaming = true;
                    frameCount = 0;
                    startTime = Date.now();
                    updateStatus('Stream started');
                    fetchFrame();
                }
            }
            
            function resetSim() {
                fetch('/reset', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => updateStatus('Simulation reset'));
            }
            
            // Auto-start
            setTimeout(() => {
                toggleStream();
            }, 500);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/reset")
async def reset_simulation():
    simulator.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    print("Starting MuJoCo server...")
    print("Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)