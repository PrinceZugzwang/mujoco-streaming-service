import os
import time
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel

import mujoco
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse


class ObjectConfig(BaseModel):
    """Configuration for adding objects to the simulation."""
    object_type: str  # "box", "sphere", "cylinder", "capsule"
    position: List[float]  # [x, y, z]
    size: List[float]  # dimensions based on type
    mass: float = 1.0
    color: List[float] = [1.0, 0.0, 0.0, 1.0]  # RGBA
    name: Optional[str] = None


class SceneConfig(BaseModel):
    """Configuration for the entire scene."""
    xml_path: Optional[str] = None
    objects: List[ObjectConfig] = []
    gravity: List[float] = [0.0, 0.0, -9.81]
    timestep: float = 0.01


class MuJoCoSimulator:
    def __init__(self, scene_config: SceneConfig = None, width: int = 320, height: int = 240):
        self.width = width
        self.height = height
        self.scene_config = scene_config or SceneConfig()
        
        # Build the simulation
        self._build_model()
        
        # Initialize camera first
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.0
        self.camera.elevation = -20
        self.camera.azimuth = 45
    
    def _build_model(self):
        """Build the MuJoCo model from config."""
        print(f"Building model with {len(self.scene_config.objects)} objects...")
        
        if self.scene_config.xml_path and os.path.exists(self.scene_config.xml_path):
            print(f"Loading model from: {self.scene_config.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(self.scene_config.xml_path)
        else:
            xml_content = self._generate_xml()
            print(f"Generated XML length: {len(xml_content)} characters")
            self.model = mujoco.MjModel.from_xml_string(xml_content)
        
        # Create new data
        self.data = mujoco.MjData(self.model)
        
        # Initialize physics state
        mujoco.mj_forward(self.model, self.data)
        
        # Recreate renderer with new model
        if hasattr(self, 'renderer'):
            del self.renderer
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        
        print(f"Model built: {self.model.nbody} bodies, {self.model.ngeom} geoms")
    
    def _generate_xml(self) -> str:
        """Generate MuJoCo XML from scene configuration."""
        gravity_str = f"{self.scene_config.gravity[0]} {self.scene_config.gravity[1]} {self.scene_config.gravity[2]}"
        
        # Start building XML string
        xml_parts = [f"""<mujoco>
    <option gravity="{gravity_str}" timestep="{self.scene_config.timestep}"/>
    <visual>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    </visual>
    <worldbody>
        <light name="top" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" size="3 3 0.05" type="plane" material="grid"/>"""]
        
        # Add all configured objects
        for i, obj in enumerate(self.scene_config.objects):
            name = obj.name or f"{obj.object_type}_{i}"
            pos_str = f"{obj.position[0]} {obj.position[1]} {obj.position[2]}"
            rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} {obj.color[3]}"
            
            # Handle different object types and their size requirements
            if obj.object_type == "box":
                size_str = f"{obj.size[0]} {obj.size[1]} {obj.size[2]}"
                geom_type = "box"
            elif obj.object_type == "sphere":
                size_str = f"{obj.size[0]}"
                geom_type = "sphere"
            elif obj.object_type == "cylinder":
                size_str = f"{obj.size[0]} {obj.size[1]}"
                geom_type = "cylinder"
            elif obj.object_type == "capsule":
                size_str = f"{obj.size[0]} {obj.size[1]}"
                geom_type = "capsule"
            else:
                continue
            
            xml_parts.append(f"""
        <body name="{name}" pos="{pos_str}">
            <joint name="{name}_joint" type="free"/>
            <geom name="{name}_geom" size="{size_str}" type="{geom_type}" rgba="{rgba_str}" mass="{obj.mass}"/>
        </body>""")
        
        # Close XML structure
        xml_parts.append("""
    </worldbody>
    <asset>
        <material name="grid" texture="grid"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
                 rgb1="0.8 0.8 0.8" rgb2="0.2 0.3 0.4"/>
    </asset>
</mujoco>""")
        
        return "".join(xml_parts)
    
    def add_object(self, obj_config: ObjectConfig) -> bool:
        """Add a new object to the simulation by rebuilding the entire model."""
        try:
            # Store current state
            old_time = self.data.time if hasattr(self, 'data') else 0
            old_camera_settings = {
                'distance': self.camera.distance,
                'elevation': self.camera.elevation,
                'azimuth': self.camera.azimuth
            } if hasattr(self, 'camera') else None
            
            # Add object to configuration
            self.scene_config.objects.append(obj_config)
            
            # Rebuild everything
            self._build_model()
            
            # Restore camera settings
            if old_camera_settings:
                self.camera.distance = old_camera_settings['distance']
                self.camera.elevation = old_camera_settings['elevation']
                self.camera.azimuth = old_camera_settings['azimuth']
            
            # Restore time
            self.data.time = old_time
            
            # Run a few simulation steps to settle physics
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            
            print(f"Successfully added {obj_config.object_type} at {obj_config.position}")
            return True
            
        except Exception as e:
            print(f"Failed to add object: {e}")
            import traceback
            traceback.print_exc()
            # Remove the object if it was added
            if self.scene_config.objects and self.scene_config.objects[-1] == obj_config:
                self.scene_config.objects.pop()
            return False
    
    def step(self):
        """Advance physics simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)
    
    def render_frame(self) -> np.ndarray:
        """Render current simulation state to RGB array."""
        # Ensure physics state is current
        mujoco.mj_forward(self.model, self.data)
        
        # Update scene and render
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
    
    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        # Run forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
    
    def get_scene_info(self) -> dict:
        """Get information about the current scene."""
        return {
            "total_bodies": self.model.nbody,
            "configured_objects": len(self.scene_config.objects),
            "simulation_time": float(self.data.time),
            "gravity": self.scene_config.gravity,
            "timestep": self.scene_config.timestep
        }


# Global simulator
xml_path = os.getenv('MUJOCO_MODEL_PATH')
initial_config = SceneConfig(xml_path=xml_path)
simulator = MuJoCoSimulator(scene_config=initial_config)

# FastAPI application
app = FastAPI(title="MuJoCo Streaming Server")


@app.get("/frame")
async def get_frame():
    """Get a single frame as JPEG."""
    frame_start = time.time()
    
    # Step physics and render
    simulator.step()
    rgb_array = simulator.render_frame()
    
    # Convert to JPEG
    image = Image.fromarray(rgb_array, mode='RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=50, optimize=False)
    
    frame_time = (time.time() - frame_start) * 1000
    
    return StreamingResponse(
        BytesIO(buffer.getvalue()),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Frame-Time": str(int(frame_time))
        }
    )


@app.post("/add_object")
async def add_object(obj_config: ObjectConfig):
    """Add a new object to the simulation."""
    try:
        success = simulator.add_object(obj_config)
        if success:
            return {
                "status": "success",
                "message": f"Added {obj_config.object_type} to simulation",
                "total_objects": len(simulator.scene_config.objects)
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add object")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/object_templates")
async def get_object_templates():
    """Get template configurations for standard objects."""
    return {
        "box": {
            "object_type": "box",
            "position": [0.0, 0.0, 1.0],
            "size": [0.1, 0.1, 0.1],
            "mass": 1.0,
            "color": [1.0, 0.0, 0.0, 1.0]
        },
        "sphere": {
            "object_type": "sphere", 
            "position": [0.0, 0.0, 1.0],
            "size": [0.1],
            "mass": 1.0,
            "color": [0.0, 1.0, 0.0, 1.0]
        },
        "cylinder": {
            "object_type": "cylinder",
            "position": [0.0, 0.0, 1.0], 
            "size": [0.05, 0.2],
            "mass": 1.0,
            "color": [0.0, 0.0, 1.0, 1.0]
        },
        "capsule": {
            "object_type": "capsule",
            "position": [0.0, 0.0, 1.0],
            "size": [0.05, 0.15],
            "mass": 1.0,
            "color": [1.0, 1.0, 0.0, 1.0]
        }
    }


@app.get("/scene_info")
async def get_scene_info():
    """Get information about the current scene."""
    return {
        "scene_config": {
            "xml_path": simulator.scene_config.xml_path,
            "num_objects": len(simulator.scene_config.objects),
            "gravity": simulator.scene_config.gravity,
            "timestep": simulator.scene_config.timestep
        },
        "simulation_info": simulator.get_scene_info()
    }


@app.post("/reset")
async def reset_simulation():
    """Reset the simulation to initial state."""
    simulator.reset()
    return {"status": "simulation reset"}


# ===== DEBUGGING ENDPOINTS =====

@app.get("/debug/positions")
async def debug_positions():
    """Debug joint positions in the simulation."""
    positions = {}
    qpos_idx = 0
    
    for i in range(simulator.model.nbody):
        body_name = simulator.model.body(i).name
        if body_name and body_name != "world":
            # Get body position from xpos (global position)
            body_pos = simulator.data.xpos[i]
            positions[body_name] = {
                "xpos": list(body_pos),  # Global position from forward kinematics
                "qpos_start_idx": qpos_idx
            }
            
            # Check if this body has a free joint
            for j in range(simulator.model.njnt):
                if simulator.model.jnt_bodyid[j] == i:
                    joint_type = simulator.model.jnt_type[j]
                    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                        positions[body_name]["qpos"] = list(simulator.data.qpos[qpos_idx:qpos_idx+7])
                        qpos_idx += 7
    
    return {
        "qpos_length": len(simulator.data.qpos),
        "body_positions": positions,
        "has_nan": bool(np.any(np.isnan(simulator.data.qpos))),
        "has_inf": bool(np.any(np.isinf(simulator.data.qpos)))
    }


@app.get("/debug/geoms")
async def debug_geoms():
    """Debug geometry information."""
    geoms = []
    for i in range(simulator.model.ngeom):
        geom = simulator.model.geom(i)
        geom_data = {
            "id": i,
            "name": simulator.model.geom(i).name,
            "type": int(geom.type),
            "size": list(geom.size[:3]),
            "rgba": list(geom.rgba),
            "bodyid": int(geom.bodyid),
            "body_name": simulator.model.body(geom.bodyid).name if geom.bodyid >= 0 else "world"
        }
        
        # Get global position of geom
        if i < len(simulator.data.geom_xpos):
            geom_data["xpos"] = list(simulator.data.geom_xpos[i])
        
        geoms.append(geom_data)
    
    return {
        "total_geoms": simulator.model.ngeom,
        "geoms": geoms
    }


@app.get("/debug/full_state")
async def debug_full_state():
    """Complete simulation state debug."""
    return {
        "model": {
            "nbody": simulator.model.nbody,
            "ngeom": simulator.model.ngeom,
            "njnt": simulator.model.njnt,
            "nq": simulator.model.nq,
            "nv": simulator.model.nv,
            "body_names": [simulator.model.body(i).name for i in range(simulator.model.nbody)]
        },
        "data": {
            "time": float(simulator.data.time),
            "qpos_shape": simulator.data.qpos.shape,
            "qvel_shape": simulator.data.qvel.shape,
            "has_nan_qpos": bool(np.any(np.isnan(simulator.data.qpos))),
            "has_inf_qpos": bool(np.any(np.isinf(simulator.data.qpos))),
            "qpos_sample": list(simulator.data.qpos[:min(14, len(simulator.data.qpos))])
        },
        "renderer": {
            "width": simulator.renderer.width,
            "height": simulator.renderer.height
        },
        "camera": {
            "distance": float(simulator.camera.distance),
            "elevation": float(simulator.camera.elevation),
            "azimuth": float(simulator.camera.azimuth),
            "lookat": list(simulator.camera.lookat) if hasattr(simulator.camera, 'lookat') else None
        },
        "objects_config": [
            {
                "type": obj.object_type,
                "position": obj.position,
                "size": obj.size,
                "color": obj.color
            } for obj in simulator.scene_config.objects
        ]
    }


@app.get("/debug/xml")
async def debug_xml():
    """See the actual XML being generated."""
    try:
        xml_content = simulator._generate_xml()
        return {
            "xml_generation": "success",
            "xml_content": xml_content,
            "xml_length": len(xml_content),
            "num_objects": len(simulator.scene_config.objects)
        }
    except Exception as e:
        return {
            "xml_generation": "failed",
            "error": str(e)
        }


@app.get("/debug/render_test")
async def debug_render_test():
    """Test rendering with different camera angles."""
    results = []
    original_settings = {
        'distance': simulator.camera.distance,
        'elevation': simulator.camera.elevation,
        'azimuth': simulator.camera.azimuth
    }
    
    # Test different camera positions
    test_positions = [
        {'distance': 1.0, 'elevation': -45, 'azimuth': 0},
        {'distance': 3.0, 'elevation': -10, 'azimuth': 90},
        {'distance': 5.0, 'elevation': -30, 'azimuth': 180},
        {'distance': 2.0, 'elevation': -90, 'azimuth': 45},  # Top-down view
    ]
    
    for pos in test_positions:
        simulator.camera.distance = pos['distance']
        simulator.camera.elevation = pos['elevation']
        simulator.camera.azimuth = pos['azimuth']
        
        # Render and check for non-zero pixels
        rgb = simulator.render_frame()
        non_zero = np.count_nonzero(rgb)
        total_pixels = rgb.shape[0] * rgb.shape[1] * rgb.shape[2]
        
        results.append({
            "camera": pos,
            "non_zero_pixels": non_zero,
            "total_pixels": total_pixels,
            "percentage": f"{(non_zero / total_pixels * 100):.2f}%",
            "mean_brightness": float(np.mean(rgb))
        })
    
    # Restore original camera settings
    simulator.camera.distance = original_settings['distance']
    simulator.camera.elevation = original_settings['elevation']
    simulator.camera.azimuth = original_settings['azimuth']
    
    return {"render_tests": results}


@app.get("/debug/force_test_scene")
async def force_test_scene():
    """Force load a test scene with multiple objects."""
    test_objects = [
        ObjectConfig(object_type="box", position=[0, 0, 1.0], size=[0.3, 0.3, 0.3], 
                    color=[1.0, 0.0, 0.0, 1.0], mass=1.0),
        ObjectConfig(object_type="sphere", position=[0.5, 0, 1.5], size=[0.2], 
                    color=[0.0, 1.0, 0.0, 1.0], mass=0.5),
        ObjectConfig(object_type="cylinder", position=[-0.5, 0, 2.0], size=[0.1, 0.3], 
                    color=[0.0, 0.0, 1.0, 1.0], mass=0.8),
    ]
    
    # Clear existing objects and add test objects
    simulator.scene_config.objects.clear()
    
    success_count = 0
    for obj in test_objects:
        if simulator.add_object(obj):
            success_count += 1
    
    return {
        "status": "test scene loaded",
        "objects_added": success_count,
        "total_objects": len(simulator.scene_config.objects)
    }


@app.get("/debug/physics_state")
async def debug_physics_state():
    """Check physics simulation state."""
    # Run a few physics steps
    initial_qpos = simulator.data.qpos.copy()
    
    for _ in range(10):
        mujoco.mj_step(simulator.model, simulator.data)
    
    qpos_changed = not np.allclose(initial_qpos, simulator.data.qpos)
    
    return {
        "physics_running": qpos_changed,
        "qpos_changed": qpos_changed,
        "time": float(simulator.data.time),
        "energy": {
            "kinetic": float(simulator.data.energy[0]),
            "potential": float(simulator.data.energy[1])
        },
        "qpos_diff": float(np.max(np.abs(simulator.data.qpos - initial_qpos))) if len(initial_qpos) > 0 else 0
    }


@app.get("/set_camera")
async def set_camera_get(distance: float = 2.0, elevation: float = -20, azimuth: float = 45):
    """Adjust camera position via GET request."""
    simulator.camera.distance = distance
    simulator.camera.elevation = elevation
    simulator.camera.azimuth = azimuth
    return {
        "status": "camera updated",
        "distance": distance,
        "elevation": elevation, 
        "azimuth": azimuth
    }


@app.get("/health")
async def health_check():
    """Simple health check."""
    return {
        "status": "healthy", 
        "mujoco_gl": os.getenv('MUJOCO_GL', 'auto'),
        "total_objects": len(simulator.scene_config.objects),
        "model_bodies": simulator.model.nbody,
        "model_geoms": simulator.model.ngeom
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Complete web interface for MuJoCo streaming with object addition."""
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
            }
            .container {
                display: flex;
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .stream-section {
                flex: 1;
                text-align: center;
            }
            .controls-section {
                width: 350px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                height: fit-content;
            }
            .stream-container {
                border: 2px solid #333;
                border-radius: 8px;
                display: inline-block;
                background-color: #000;
                margin-bottom: 15px;
            }
            #stream {
                display: block;
                width: 320px;
                height: 240px;
            }
            button {
                padding: 8px 16px;
                margin: 5px;
                font-size: 14px;
                border: none;
                border-radius: 4px;
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }
            button:hover { background-color: #0056b3; }
            button.success { background-color: #28a745; }
            button.danger { background-color: #dc3545; }
            button.warning { background-color: #ffc107; color: #000; }
            .status {
                margin: 10px 0;
                padding: 10px;
                background-color: #fff;
                border-radius: 4px;
                font-family: monospace;
                font-size: 12px;
                border: 1px solid #ddd;
            }
            .object-form {
                margin: 15px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f9f9f9;
            }
            .form-group {
                margin: 10px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            input, select {
                width: 100%;
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 12px;
                box-sizing: border-box;
            }
            .input-row {
                display: grid;
                gap: 5px;
            }
            .input-row.three { grid-template-columns: 1fr 1fr 1fr; }
            .input-row.two { grid-template-columns: 1fr 1fr; }
            .input-row.four { grid-template-columns: 1fr 1fr 1fr 1fr; }
            .quick-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 5px;
                margin: 10px 0;
            }
            .quick-buttons button {
                margin: 0;
                font-size: 12px;
                padding: 6px;
            }
            h3 {
                margin-top: 0;
                color: #333;
            }
            .debug-section {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .debug-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 5px;
            }
            .debug-buttons button {
                font-size: 11px;
                padding: 5px;
                background-color: #6c757d;
            }
        </style>
    </head>
    <body>
        <h1>MuJoCo Live Simulation with Dynamic Objects</h1>
        
        <div class="container">
            <div class="stream-section">
                <div class="stream-container">
                    <img id="stream" alt="MuJoCo Stream">
                </div>
                
                <div>
                    <button onclick="toggleStream()" id="streamBtn">Start Stream</button>
                    <button onclick="resetSim()" class="danger">Reset Simulation</button>
                    <button onclick="getSceneInfo()">Scene Info</button>
                    <button onclick="loadTestScene()" class="warning">Load Test Scene</button>
                </div>
                
                <div class="status" id="status">Ready to start...</div>
            </div>
            
            <div class="controls-section">
                <h3>Add Objects</h3>
                
                <div class="quick-buttons">
                    <button onclick="addQuickObject('box', [1,0,0,1])" style="background:#dc3545;">Red Box</button>
                    <button onclick="addQuickObject('sphere', [0,1,0,1])" style="background:#28a745;">Green Sphere</button>
                    <button onclick="addQuickObject('cylinder', [0,0,1,1])" style="background:#007bff;">Blue Cylinder</button>
                    <button onclick="addQuickObject('capsule', [1,1,0,1])" style="background:#ffc107; color:#000;">Yellow Capsule</button>
                </div>
                
                <div class="object-form">
                    <div class="form-group">
                        <label>Object Type:</label>
                        <select id="objectType" onchange="updateSizeInputs()">
                            <option value="box">Box</option>
                            <option value="sphere">Sphere</option>
                            <option value="cylinder">Cylinder</option>
                            <option value="capsule">Capsule</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Position (X, Y, Z):</label>
                        <div class="input-row three">
                            <input type="number" id="posX" value="0" step="0.1" placeholder="X">
                            <input type="number" id="posY" value="0" step="0.1" placeholder="Y">
                            <input type="number" id="posZ" value="1.5" step="0.1" placeholder="Z">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label id="sizeLabel">Size (W, H, D):</label>
                        <div class="input-row three" id="sizeInputs">
                            <input type="number" id="size1" value="0.1" step="0.01" min="0.01" placeholder="W">
                            <input type="number" id="size2" value="0.1" step="0.01" min="0.01" placeholder="H">
                            <input type="number" id="size3" value="0.1" step="0.01" min="0.01" placeholder="D">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Mass:</label>
                        <input type="number" id="mass" value="1.0" step="0.1" min="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label>Color (R, G, B, A):</label>
                        <div class="input-row four">
                            <input type="number" id="colorR" value="1.0" step="0.1" min="0" max="1" placeholder="R">
                            <input type="number" id="colorG" value="0.0" step="0.1" min="0" max="1" placeholder="G">
                            <input type="number" id="colorB" value="0.0" step="0.1" min="0" max="1" placeholder="B">
                            <input type="number" id="colorA" value="1.0" step="0.1" min="0" max="1" placeholder="A">
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
                        <button onclick="addObject()" class="success">Add Object</button>
                        <button onclick="loadTemplate()">Load Template</button>
                    </div>
                </div>
                
                <div class="debug-section">
                    <h3>Debug Tools</h3>
                    <div class="debug-buttons">
                        <button onclick="debugPositions()">Check Positions</button>
                        <button onclick="debugGeoms()">Check Geoms</button>
                        <button onclick="debugFullState()">Full State</button>
                        <button onclick="debugPhysics()">Physics State</button>
                        <button onclick="debugRender()">Render Test</button>
                        <button onclick="debugXML()">View XML</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let streaming = false;
            let frameCount = 0;
            let startTime = Date.now();
            let totalFrameTime = 0;
            
            const img = document.getElementById('stream');
            const status = document.getElementById('status');
            const streamBtn = document.getElementById('streamBtn');
            
            function updateStatus(msg) {
                status.textContent = new Date().toLocaleTimeString() + ': ' + msg;
                console.log(msg);
            }
            
            function fetchFrame() {
                const requestStart = Date.now();
                const timestamp = Date.now();
                img.src = '/frame?' + timestamp;
                
                img.onload = function() {
                    const requestTime = Date.now() - requestStart;
                    totalFrameTime += requestTime;
                    frameCount++;
                    
                    if (frameCount % 30 === 0) {
                        const elapsed = Date.now() - startTime;
                        const actualFPS = (frameCount / elapsed * 1000).toFixed(1);
                        const avgFrameTime = (totalFrameTime / frameCount).toFixed(0);
                        updateStatus(`FPS: ${actualFPS}, Avg frame time: ${avgFrameTime}ms`);
                    }
                    
                    if (streaming) {
                        const targetInterval = 33; // 30 FPS
                        const nextDelay = Math.max(1, targetInterval - requestTime);
                        setTimeout(fetchFrame, nextDelay);
                    }
                };
                
                img.onerror = function() {
                    updateStatus('Frame loading failed');
                    if (streaming) {
                        setTimeout(fetchFrame, 100);
                    }
                };
            }
            
            function toggleStream() {
                if (streaming) {
                    streaming = false;
                    streamBtn.textContent = 'Start Stream';
                    streamBtn.style.backgroundColor = '#007bff';
                    updateStatus('Stream stopped');
                } else {
                    streaming = true;
                    frameCount = 0;
                    startTime = Date.now();
                    totalFrameTime = 0;
                    streamBtn.textContent = 'Stop Stream';
                    streamBtn.style.backgroundColor = '#dc3545';
                    updateStatus('Stream started');
                    fetchFrame();
                }
            }
            
            function resetSim() {
                fetch('/reset', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => updateStatus('Simulation reset'))
                    .catch(error => updateStatus('Reset failed: ' + error.message));
            }
            
            function updateSizeInputs() {
                const type = document.getElementById('objectType').value;
                const sizeLabel = document.getElementById('sizeLabel');
                const sizeInputs = document.getElementById('sizeInputs');
                
                if (type === 'sphere') {
                    sizeLabel.textContent = 'Size (Radius):';
                    sizeInputs.className = 'input-row';
                    sizeInputs.innerHTML = '<input type="number" id="size1" value="0.1" step="0.01" min="0.01" placeholder="Radius">';
                } else if (type === 'cylinder' || type === 'capsule') {
                    sizeLabel.textContent = 'Size (Radius, Height):';
                    sizeInputs.className = 'input-row two';
                    sizeInputs.innerHTML = `
                        <input type="number" id="size1" value="0.05" step="0.01" min="0.01" placeholder="Radius">
                        <input type="number" id="size2" value="0.2" step="0.01" min="0.01" placeholder="Height">
                    `;
                } else {
                    sizeLabel.textContent = 'Size (W, H, D):';
                    sizeInputs.className = 'input-row three';
                    sizeInputs.innerHTML = `
                        <input type="number" id="size1" value="0.1" step="0.01" min="0.01" placeholder="W">
                        <input type="number" id="size2" value="0.1" step="0.01" min="0.01" placeholder="H">
                        <input type="number" id="size3" value="0.1" step="0.01" min="0.01" placeholder="D">
                    `;
                }
            }
            
            function addObject() {
                const objectConfig = {
                    object_type: document.getElementById('objectType').value,
                    position: [
                        parseFloat(document.getElementById('posX').value),
                        parseFloat(document.getElementById('posY').value),
                        parseFloat(document.getElementById('posZ').value)
                    ],
                    size: [
                        parseFloat(document.getElementById('size1').value),
                        parseFloat(document.getElementById('size2')?.value || 0),
                        parseFloat(document.getElementById('size3')?.value || 0)
                    ].filter(x => x > 0),
                    mass: parseFloat(document.getElementById('mass').value),
                    color: [
                        parseFloat(document.getElementById('colorR').value),
                        parseFloat(document.getElementById('colorG').value),
                        parseFloat(document.getElementById('colorB').value),
                        parseFloat(document.getElementById('colorA').value)
                    ]
                };
                
                fetch('/add_object', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(objectConfig)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateStatus(`Added ${objectConfig.object_type}! Total: ${data.total_objects}`);
                    } else {
                        updateStatus('Failed to add object: ' + (data.detail || 'Unknown error'));
                    }
                })
                .catch(error => updateStatus('Error: ' + error.message));
            }
            
            function addQuickObject(type, color) {
                const randomPos = () => (Math.random() - 0.5) * 1.0;
                
                let size, mass;
                if (type === 'sphere') {
                    size = [0.08];
                    mass = 0.5;
                } else if (type === 'cylinder' || type === 'capsule') {
                    size = [0.05, 0.15];
                    mass = 0.8;
                } else {
                    size = [0.1, 0.1, 0.1];
                    mass = 1.0;
                }
                
                const objectConfig = {
                    object_type: type,
                    position: [randomPos(), randomPos(), 1.5 + Math.random()],
                    size: size,
                    mass: mass,
                    color: color
                };
                
                fetch('/add_object', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(objectConfig)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateStatus(`Quick added ${type}! Total: ${data.total_objects}`);
                    } else {
                        updateStatus('Failed: ' + (data.detail || 'Unknown error'));
                    }
                })
                .catch(error => updateStatus('Error: ' + error.message));
            }
            
            function loadTemplate() {
                fetch('/object_templates')
                    .then(response => response.json())
                    .then(templates => {
                        const type = document.getElementById('objectType').value;
                        const template = templates[type];
                        
                        if (template) {
                            document.getElementById('posX').value = template.position[0];
                            document.getElementById('posY').value = template.position[1];
                            document.getElementById('posZ').value = template.position[2];
                            
                            document.getElementById('size1').value = template.size[0];
                            if (template.size[1] !== undefined && document.getElementById('size2')) 
                                document.getElementById('size2').value = template.size[1];
                            if (template.size[2] !== undefined && document.getElementById('size3')) 
                                document.getElementById('size3').value = template.size[2];
                            
                            document.getElementById('mass').value = template.mass;
                            document.getElementById('colorR').value = template.color[0];
                            document.getElementById('colorG').value = template.color[1];
                            document.getElementById('colorB').value = template.color[2];
                            document.getElementById('colorA').value = template.color[3];
                            
                            updateStatus(`Loaded ${type} template`);
                        }
                    })
                    .catch(error => updateStatus('Template load failed: ' + error.message));
            }
            
            function getSceneInfo() {
                fetch('/scene_info')
                    .then(response => response.json())
                    .then(data => {
                        const info = `Objects: ${data.scene_config.num_objects}, Bodies: ${data.simulation_info.total_bodies}, Time: ${data.simulation_info.simulation_time.toFixed(2)}s`;
                        updateStatus(info);
                    })
                    .catch(error => updateStatus('Scene info failed: ' + error.message));
            }
            
            function loadTestScene() {
                fetch('/debug/force_test_scene')
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(`Test scene loaded: ${data.objects_added} objects`);
                    })
                    .catch(error => updateStatus('Test scene failed: ' + error.message));
            }
            
            // Debug functions
            function debugPositions() {
                fetch('/debug/positions')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Position Debug:', data);
                        updateStatus(`Bodies: ${Object.keys(data.body_positions).length}, NaN: ${data.has_nan}`);
                        alert(JSON.stringify(data, null, 2));
                    });
            }
            
            function debugGeoms() {
                fetch('/debug/geoms')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Geom Debug:', data);
                        updateStatus(`Geoms: ${data.total_geoms}`);
                        alert(JSON.stringify(data, null, 2));
                    });
            }
            
            function debugFullState() {
                fetch('/debug/full_state')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Full State:', data);
                        updateStatus('Full state logged to console');
                        alert(JSON.stringify(data, null, 2));
                    });
            }
            
            function debugPhysics() {
                fetch('/debug/physics_state')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Physics State:', data);
                        updateStatus(`Physics running: ${data.physics_running}`);
                        alert(JSON.stringify(data, null, 2));
                    });
            }
            
            function debugRender() {
                fetch('/debug/render_test')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Render Test:', data);
                        updateStatus('Render test complete - check console');
                        alert(JSON.stringify(data, null, 2));
                    });
            }
            
            function debugXML() {
                fetch('/debug/xml')
                    .then(response => response.json())
                    .then(data => {
                        console.log('XML:', data.xml_content);
                        updateStatus(`XML generated: ${data.xml_length} chars`);
                        alert(data.xml_content);
                    });
            }
            
            // Initialize
            updateSizeInputs();
            
            // Auto-start stream after short delay
            setTimeout(() => {
                toggleStream();
            }, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    
    print("Starting MuJoCo streaming server...")
    print("Environment:")
    print(f"  MUJOCO_GL: {os.getenv('MUJOCO_GL', 'auto-detect')}")
    print(f"  MUJOCO_MODEL_PATH: {os.getenv('MUJOCO_MODEL_PATH', 'not set')}")
    print(f"  Total configured objects: {len(simulator.scene_config.objects)}")
    print(f"  Total bodies in simulation: {simulator.model.nbody}")
    print()
    print("Server will be available at:")
    print("  Local: http://localhost:8000")
    print("  Stream endpoint: http://localhost:8000/frame")
    print("  Health check: http://localhost:8000/health")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
