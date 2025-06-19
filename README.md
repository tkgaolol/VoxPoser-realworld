# Robot Manipulation System

A language-guided robotic manipulation system that combines computer vision, natural language processing, and motion planning.

## 📋 Prerequisites

### Required Libraries
```python
ultralytics        # YOLO object detection
torch             # Deep learning framework
opencv-python     # Computer vision
pyrealsense2      # RealSense camera interface
jupyter           # Notebook interface
openai            # LLM integration
plotly            # Data visualization
transforms3d      # 3D transformations
open3d            # 3D data processing 
numpy==1.26.4     # Numerical computing
pyserial            # Serial communication
transformers      
accelerate
```
you may need to run 
```bash
conda install -c conda-forge libstdcxx-ng 
```
if you encounter 
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

## 🛠️ Setup Instructions
to be implemented
### Hardware Setup
1. Robot arm (TCP connection)
2. Gripper (Serial connection)
3. Depth Cameras:
   - RealSense camera
   - Orbbec Femto Mega camera (optional for multi-view)
4. Working platform

### Initial Configuration
1. Set up OpenAI API key
2. Install required submodules:
   ```bash
   # Clone submodules for vision components
   git submodule add https://github.com/hkchengrex/XMem.git real/toolbox/perceptron/XMem
   git submodule add https://github.com/orbbec/pyorbbecsdk.git real/toolbox/perceptron/femto_mega/pyorbbecsdk
   git submodule update --init --recursive
   ```
3. Camera Setup:
   - For RealSense: Install `pyrealsense2` package
   - For Orbbec Femto Mega: Install provided wheel file:
     ```bash
     pip install real/toolbox/perceptron/femto_mega/pyorbbecsdk-1.3.1-cp310-cp310-linux_x86_64.whl
     ```
4. Perform camera-to-robot calibration:
   ```bash
   python cal_Transform_mat.py
   ```
   - Press `Space` to capture current frame
   - Move robot arm to different positions (10 or more captures needed)
   - Press `Esc` or `Q` to exit
   - Copy the resulting transform matrix to `cam2robot.py`

   Note: For multi-camera setup, calibration needs to be performed for each camera.


## 📁 Code Structure
```
.
├── .vscode/                    # VS Code configuration
│   └── launch.json            # Debug configurations
├── real/                      # Main project implementation
│   ├── configs/               # Configuration files
│   ├── model_weight/          # Pre-trained model weights
│   ├── prompts/               # LLM prompt templates
│   ├── toolbox/               # Core functionality modules
│   │   ├── perceptron/       # Vision and perception tools
│   │   │   ├── XMem/         # Video object segmentation
│   │   │   ├── femto_mega/   # Orbbec camera SDK
│   │   │   ├── calibration/  # Camera-robot calibration
│   │   │   └── ...          # Other perception tools
│   │   ├── real_env.py       # Environment interface
│   │   ├── gripper.py        # Gripper control
│   │   └── tcp_connection.py # Robot communication
│   ├── LMP.py                # Language Model Programs
│   ├── interfaces.py         # System interfaces
│   ├── controllers.py        # Motion controllers
│   ├── planners.py          # Motion planning
│   ├── real.py              # Main execution
│   └── visualizers.py       # Visualization tools
├── setup.py                  # Package configuration
└── README.md                # Project documentation
```


## 🔄 System Pipeline

### 1. Natural Language Command Processing
- Input example: "grab the mouse"
- Command parsing and task decomposition
- Spatial relationship interpretation

### 2. Perception System
1. **Object Detection**
   - YOLO-based RGB object detection
   - RealSense depth information acquisition
   - 3D coordinate conversion using:
     - RGB-D fusion
     - Camera intrinsics
     - Calibration transform matrix

2. **Advanced Vision Processing**
   - OWL-ViT for open-vocabulary detection
   - Segment Anything for mask generation
   - XMEM for mask tracking
   - Point cloud reconstruction

### 3. Motion Planning

#### Language Model Programs (LMP)
1. **Planner**: High-level command → Sub-tasks
2. **Composer**: Sub-tasks → Motion parameters
3. **Parser**: Object and spatial relationship interpretation

#### Value Maps
- Affordance mapping
- Obstacle avoidance
- End-effector velocity control
- Rotation constraints
- Gripper action planning

#### Trajectory Planning
- Voxel-based representation (100×100×100×k)
- Euclidean distance transform for affordance
- Gaussian filtering for avoidance
- Cost optimization (2:1 affordance-avoidance weighting)

### 4. Execution and Control
1. Coordinate transformation
2. Real-time monitoring:
   - Collision detection
   - Gripper state
   - End-effector positioning
3. Dynamic replanning
4. Push motion parameterization:
   - Contact points
   - Push direction
   - Push distance




## 📝 Documentation
### LLMs and Prompting
1. 使用自己生成的代码递归调用 LLM
2. 其中每个语言模型程序 (LMP) 负责一个独特的功能（例如，处理感知调用）
3. 对于每个 LMP，我们将 5-20 个示例查询和相应的响应作为提示的一部分。
4. planner 将用户指令作为输入（例如，"打开抽屉"）并输出一系列子任务
5. composer 接收子任务并调用具有详细语言参数化的相关价值图 LMP

### VLMs and Perception
1. 调用开放词汇检测器 OWL-ViT 以获得边界框
2. 将其输入到 Segment Anything 中以获得mask
3. 使用视频跟踪器 XMEM 跟踪mask
4. 跟踪的mask与 RGB-D 信息 会一起使用以重建对象/部分点云

### Value Map Composition
1. affordance, avoidance, endeffector velocity, end-effector rotation, and gripper action
2. 每种类型使用不同的 LMP：接收指令并输出形状为 (100, 100, 100, k) 的体素图，其中 k 对于每个价值图都不同（例如，可供性和避免性的 k = 1，因为它指定成本，而旋转的 k = 4，因为它指定 SO(3)）
3. 欧几里得距离变换应用于affordance，并将高斯滤波器应用于avoidance

### Motion Planner
1. consider only affordance and avoidance maps in the planner optimization
2. greedy search找到一系列无碰撞的末端执行器位置
3. 在每个位置，我们通过剩余的值图（例如，旋转图、速度图）强制其他参数化
4. The cost map used by the motion planner is computed as the negative of the weighted sum
of normalized affordance and avoidance maps with weights 2 and 1

### Dynamics Model
1. 不使用环境动力学模型（即，假设场景是静态的）
2. 在每一步都重新规划以考虑最新的观察结果
3. 只研究由接触点、推动方向和推动距离参数化的平面推动模型



## 🔍 TODO
1. 可以看一下文章最后的api 和 prompts的写法
2. 具体实现要看文章第四部分开头
3. 联合ros一起搞吗？
4. 全臂避障规划
5. 记得准备板子 把工作平台分割好
6. 抓取姿势,夹爪夹多少，力给多少 FoundationPoseg
7. 多整个深度相机？用kinect，合并点云，标定。不然他提取的目标位置是表面不是中心
8. 可能要重新看一下标定怎么做，因为夹爪的位置不是机械臂末端的位置
9. 测试所有demo 换个物体能检测到的

## Implementation notes
1. cv2.imshow will freeze if you import av simultaneously(because ffmpeg used by av and opencv is not compatible)
2. numpy 1.26.4 is required for the femto_mega camera

## Acknowledgements
This project is built on top of [VoxPoser](https://github.com/huangwl18/VoxPoser).
