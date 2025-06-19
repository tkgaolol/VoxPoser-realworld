# Real World deployment for Voxposer

## NOT Yet Finished
1. Gripper rotation and position (current implementation is fix the position before running a task)
2. Robust testing is not stable (if I move the target while the robot arm is executing a task, it will not update to the new position)
3. whole-arm obstacle avoidance planning
4. multi depth camera? (current implementation will only get the center point from one surface)

## 🛠️ Setup Instructions
### Hardware Setup
1. Robot arm (TCP connection)
2. Gripper (Serial connection)
3. Depth Cameras (RealSense camera) 
4. Workspace (remember to strict the workspace bounds for robot arm inside real_env.py)<br>

You should use scripts under src/toolbox to test the connection of external devices


### Initial Configuration
1. Obtain an [OpenAI API](https://openai.com/blog/openai-api) key, and put it inside config.ini

2. Install required submodules:
   ```bash
   # Clone submodules for vision components(XMem)
   git submodule update --init --recursive
   ```

3. Create a conda environment:
   ```Shell
   conda create -n voxposer-realworld-env python=3.10
   conda activate voxposer-realworld-env
   ```


4. Install dependencies:
   ```Shell
   pip install -r requirements.txt
   ```

   you may need to run 
   ```bash
   conda install -c conda-forge libstdcxx-ng 
   ```
   if you encounter the following error
   ```bash
   libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
   libGL error: failed to load driver: iris
   libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
   libGL error: failed to load driver: iris
   libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
   libGL error: failed to load driver: swrast
   [Open3D WARNING] GLFW Error: GLX: Failed to create context: GLXBadFBConfig
   [Open3D WARNING] Failed to create window
   [Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.
   ```

5. Download following models from [ultralytics](https://docs.ultralytics.com/models) and [XMem](https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1)
   - sam2.1_b.pt
   - yolo11-seg.pt
   - yolo11x.pt
   - XMem.pth


6. Perform camera-to-robot calibration:
   ```bash
   python src/toolbox/perceptron/d435i/calibration/cal_transform_mat.py
   ```
   - Press `Space` to capture the current frame and process a calibration sample.
   - Move the robot arm to different positions (10 or more captures recommended) and repeat the last step.
   - Press `d` to delete the last collected sample if needed.
   - Press `r` to reset all collected calibration data.
   - Press `Esc` to finish calibration and compute the transformation matrix.
   - Copy the resulting transform matrix to `cam2robot.py` and test the accuracy.
   - Replace the transform matrix to `run.py`

   Note: For multi-camera setup, calibration needs to be performed for each camera.

7. Start to play<br>
   
   You may need to adjust the code acoording the devices you used
   ```bash
   python src/run.py
   ```

## 📁 Code Structure
```
.
├── .vscode/                    # VS Code configuration
│   └── launch.json             # Debug configurations
├── media/
├── src/                        # Main project implementation
│   ├── configs/                # Configuration files
│   ├── model_weight/           # Pre-trained model weights
│   ├── prompts/                # LLM prompt templates
│   │   └── rlbench/            # RLBench prompt templates
│   ├── toolbox/                # Core functionality modules
│   │   ├── perceptron/         # Vision and perception tools
│   │   │   ├── XMem/           # Video object segmentation
│   │   │   ├── d435i/          # RealSense camera tools
│   │   │   │   └── calibration/ # Camera-robot calibration
│   │   │   └── ...             # Other perception tools
│   │   ├── my_prompt/          # Custom prompt templates
│   │   ├── real_env.py         # Environment interface
│   ├── envs/                   # Environment definitions
│   └── run.py                  # Main execution entry point
├── requirements.txt            # Python dependencies
├── config.ini                  # API and config keys
└── README.md                   # Project documentation
```







## Acknowledgements
This project is built on top of [VoxPoser](https://github.com/huangwl18/VoxPoser).
