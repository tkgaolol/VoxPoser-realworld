import numpy as np
import open3d as o3d
import time
import sys
from toolbox.tcpsock import TCP_SOCKET
from toolbox.perceptron.camera_tracker_d435i import CameraTracker
from toolbox.perceptron.d435i.calibration.cam2robot import CoordinateTransformer
from toolbox.gripper import GripperController
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
import cv2

class RealRobotEnv(TCP_SOCKET):
    def __init__(self, serial_port='COM3', visualizer=None, instruction=None, T_camera_to_base=None, debug=False):
        """
        Initializes the RealRobotEnv environment.
        
        Args:
            serial_port: Serial port for gripper control.
            visualizer: Visualization interface, optional
            T_camera_to_base: Transformation matrix from camera to robot base
            debug: Debug mode
        """
        TCP_SOCKET.__init__(self)
        if debug:
            self.robot_arm_server_ip = '127.0.0.1'
            self.gripper = None
        else:
            self.robot_arm_server_ip = '192.168.0.10'
            self.gripper = GripperController(serial_port)

        self.robot_arm_server_port = 10003
        self.robot_arm_server_address = (self.robot_arm_server_ip, self.robot_arm_server_port)

        self.visualizer = visualizer
        self.message = instruction
        self.debug = debug

        self.cam2robot = CoordinateTransformer(T_camera_to_base)
        
        # # Define workspace bounds (in meters)
        # workspace_min = self.cam2robot.robot_to_camera(np.array([-320, -650, 60]))
        # workspace_max = self.cam2robot.robot_to_camera(np.array([30, -350, 250]))

        # # make sure min is less than max
        # self.workspace_bounds_min = np.minimum(workspace_min, workspace_max)
        # self.workspace_bounds_max = np.maximum(workspace_min, workspace_max)
        
        # self.workspace_bounds_min = self.workspace_bounds_min.astype(int)
        # self.workspace_bounds_max = self.workspace_bounds_max.astype(int)

        self.workspace_bounds_min = np.array([-600, -800, 0])
        self.workspace_bounds_max = np.array([30, -100, 450])

        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        
        self.tracker = CameraTracker()

        self.init_obs = self.get_ee_pose()
        self.latest_obs = self.init_obs
        self.latest_reward = 0
        self.latest_terminate = False
        self.latest_action = None
        self.objects_on_table = self.get_object_names()
        self._update_visualizer()

        self.get_init_pos()
        self.first_flag_dict = dict()
        
    def get_object_names(self):
        """Returns list of detected object names"""
        return self.tracker.get_current_objects()
    
    def load_task(self, target):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        # self._reset_task_variables()
        # if isinstance(task, str):
            # task = getattr(tasks, task)
        # self.task = self.rlbench_env.get_task(task)
        # self.arm_mask_ids = [obj.get_handle() for obj in self.task._robot.arm.get_objects_in_tree(exclude_base=False)]
        # self.gripper_mask_ids = [obj.get_handle() for obj in self.task._robot.gripper.get_objects_in_tree(exclude_base=False)]
        # self.robot_mask_ids = self.arm_mask_ids + self.gripper_mask_ids
        # self.obj_mask_ids = [obj.get_handle() for obj in self.task._task.get_base().get_objects_in_tree(exclude_base=False)]
        # # store (object name <-> object id) mapping for relevant task objects
        # try:
        #     name_mapping = self.task_object_names[self.task.get_name()]
        # except KeyError:
        #     raise KeyError(f'Task {self.task.get_name()} not found in "envs/task_object_names.json" (hint: make sure the task and the corresponding object names are added to the file)')
        # exposed_names = [names[0] for names in name_mapping]
        # internal_names = [names[1] for names in name_mapping]
        # scene_objs = self.task._task.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE,
        #                                                               exclude_base=False,
        #                                                               first_generation_only=False)
        # for scene_obj in scene_objs:
        #     if scene_obj.get_name() in internal_names:
        #         exposed_name = exposed_names[internal_names.index(scene_obj.get_name())]
        #         self.name2ids[exposed_name] = [scene_obj.get_handle()]
        #         self.id2name[scene_obj.get_handle()] = exposed_name
        #         for child in scene_obj.get_objects_in_tree():
        #             self.name2ids[exposed_name].append(child.get_handle())
        #             self.id2name[child.get_handle()] = exposed_name
        self.targets = target
        
    def get_3d_obs_by_name(self, name):
        """Get 3D position of specified object"""
        target = [name]
        dict_objects = self.tracker.get_latest_objects(target)
        self.tracker.process_3d()
        count = 0
        while name not in dict_objects or dict_objects[name] is None or dict_objects[name]['center3d'] is None:
            dict_objects = self.tracker.get_latest_objects(target)
            self.tracker.process_3d()
            count += 1
            if count > 5:
                print(f"[INFO]: Restarting tracker for {name} count: {count}")
                self.tracker.get_latest_objects(target, restart=True)
                count = 0
            time.sleep(1)

        # save image
        if name not in self.first_flag_dict.keys():
            image = self.tracker.latest_color_image.copy()
            mask_2d = dict_objects[name]['mask2d']
            image[mask_2d > 0] = image[mask_2d > 0] * 0.7 + np.array((255, 255, 0), dtype=np.uint8) * 0.3
            cv2.imwrite(f'{name}_detection.png',image)
            self.first_flag_dict[name] = False

        center_3d = dict_objects[name]['center3d'].reshape(1, 3)
        center_3d = self.cam2robot.camera_to_robot(center_3d)

        # Ensure the z-coordinate is above a minimum threshold
        if center_3d[:, 2] < 45:
            center_3d[:, 2] = 45

        normal = dict_objects[name]['normal'].reshape(1, 3)
        
        return center_3d, normal
    
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Get 3D positions and colors of all objects in scene.
        """
        dict_objects = self.tracker.get_latest_objects(self.objects_on_table)
        while self.tracker.latest_pointcloud is None:
            self.tracker.process_3d()
            time.sleep(1)
        points = self.tracker.latest_pointcloud.reshape(-1, 3)
        colors = self.tracker.latest_color_image
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
        colors = colors.reshape(-1, 3)

        points = self.cam2robot.camera_to_robot(points)

        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        if len(points) == 0:
            return np.zeros((1,3)), np.zeros((1,3), dtype=np.uint8)
            
        # Voxel downsample using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        
        points = np.asarray(pcd_downsampled.points)
        colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)
        return points, colors
    
    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        # TODO: reset task
        # assert self.task is not None, "Please load a task first"
        # self.task.sample_variation()
        # descriptions, obs = self.task.reset()
        print("[INFO]: Resetting environment")
        descriptions = self.message
        obs = self.get_ee_pose()
        obs = self._process_obs(obs)
        self.init_obs = obs
        self.latest_obs = obs
        self._update_visualizer()
        return descriptions, obs
    
    def apply_action(self, action):
    # def apply_action(self, action, require_transform=True):
        """
        Applies an action to the robot.
        
        Args:
            action: Array containing [x,y,z,qx,qy,qz,qw,gripper_state]
        """
        # TODO: implement action processing
        # if require_transform:
        #     xyz = action[:3]
        #     xyz = self.cam2robot.camera_to_robot(xyz)
        #     action[:3] = xyz
        action = self._process_action(action)

        # round to 2 decimal places
        action = np.round(action).astype(int)
        
        pos = action[:3]
        # rot = action[3:7] if len(action) > 3 else self.init_obs[3:]  # Default quaternion if not provided
        gripper = action[7] if len(action) > 7 else 0
        # rad2deg = 180 / np.pi
        # rpy = np.array(euler_from_quaternion(rot)) * rad2deg
        rpy = action[3:6] 
        rpy = np.round(rpy).astype(int)

        # # Send movement command to robot
        # for i in range(3):
        #     command = 'MoveRelL'
        #     nRbtID = '0'
        #     nAxisId = str(i) # 0:x 1:y 2:z
        #     if pos[i] < 0:
        #         nDirection = '1'
        #     else:
        #         nDirection = '0' # 0:反向 1:正向
        #     dDistance = str(abs(pos[i]))
        #     nToolMotion = '0'
        #     message = command+','+nRbtID+','+nAxisId+','+nDirection+','+dDistance+','+nToolMotion+',;'
            

        #     self.client_socket.sendall(message.encode())
        #     time.sleep(0.1)  # Wait for movement

        message_params = {
                        'command': 'WayPoint',
                        'nRbtID': '0',
                        'dX': str(pos[0]),
                        'dY': str(pos[1]),
                        'dZ': str(pos[2]),
                        'dRx': str(rpy[0]),
                        'dRy': str(rpy[1]),
                        'dRz': str(rpy[2]),
                        'dJ1': str(self.init_pos['dJ1']),
                        'dJ2': str(self.init_pos['dJ2']),
                        'dJ3': str(self.init_pos['dJ3']),
                        'dJ4': str(self.init_pos['dJ4']),
                        'dJ5': str(self.init_pos['dJ5']),
                        'dJ6': str(self.init_pos['dJ6']),
                        'sTcpName': 'TCP_gripper',
                        'sUcsName': 'Base',
                        'dVelocity': '50',
                        'dAcc': '90',
                        'dRadius': '0',
                        'nMoveType': '0',
                        'nIsUseJoint': '0',
                        'nIsSeek': '0',
                        'nIOBit': '0',
                        'nIOState': '0',
                        'strCmdID': '0'
                    }
        message = ','.join(message_params.values()) + ',;'

        print(f"[INFO]: Sending message: {message}")

        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        
        if not success and not self.debug:
            # TODO: handle failure in arm server
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)

        # Check if response contains "ok"
        if b'ok' not in response.lower() and not self.debug:
            print("[ERROR]: Server response does not contain 'ok'")
            print(f"Server response was: {response}")
            sys.exit(1)

        # Handle gripper if specified
        if self.gripper is not None:
            if gripper >= 1 and self.targets[0] != 'lamp':
                self.open_gripper()
            else:
                 self.close_gripper()

        # obs, reward, terminate = self.process_obs(action)
        # self.latest_obs = obs
        # self.latest_reward = reward
        # self.latest_terminate = terminate
        self.latest_obs = 0
        self.latest_reward = 0
        self.latest_terminate = False
        self.latest_action = action

        self._update_visualizer()
        
        # return obs, reward, terminate
        return 0
    
    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        # if self.latest_action is None:
        #     action = np.concatenate([pose, [self.init_obs.gripper_open]])
        # else:
        #     action = np.concatenate([pose, [self.latest_action[-1]]])

        # xyz = pose[:3]
        # xyz = self.cam2robot.camera_to_robot(xyz)
        # pose[:3] = xyz
        # self.apply_action(pose, require_transform=False)
        self.apply_action(pose)

        
    def open_gripper(self):
        """Open gripper"""
        self.gripper.set_position(1000)
        
    def close_gripper(self):
        """Close gripper"""
        # ball
        self.gripper.set_position(850)
        # lamp
        # self.gripper.set_position(0)
        
    def set_gripper_state(self, gripper_state):
        """Set gripper to specified state"""
        action = np.concatenate([self.latest_obs, [gripper_state]])
        # return self.apply_action(action, require_transform=False)
        return self.apply_action(action)
            
    def reset_to_default_pose(self):
        """Reset robot to default/home position"""
        init_pos = np.array([self.init_pos['dX'], self.init_pos['dY'], self.init_pos['dZ'],
                             self.init_pos['dRx'], self.init_pos['dRy'], self.init_pos['dRz']])
        if self.latest_action is None:
            action = np.concatenate([init_pos, [1]])
        else:
            action = np.concatenate([init_pos, [self.latest_action[-1]]])
        # return self.apply_action(action, require_transform=False)
        return self.apply_action(action)
        
    def get_ee_pose(self):
        """Get current end effector position"""
        # Send request for current position
        message = "ReadActPos,0,;"
        
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success and not self.debug:
            # TODO: handle failure in arm server
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)

        # Check if response contains "ok"
        if b'ok' not in response.lower() and not self.debug:
            print("[ERROR]: Server response does not contain 'ok'")
            print(f"Server response was: {response}")
            sys.exit(1)


        if self.gripper is not None:
            pos = self.gripper.get_position()
        else:
            pos = 1000

        if self.debug:
            return np.zeros(7)
        
        response = response.decode().split(',')
        # robot_pos = self.cam2robot.robot_to_camera(np.array([float(response[8]), float(response[9]), float(response[10])]))
        
        # return np.array([float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]), float(response[11]), float(response[12]), float(response[13]), float(pos)])
        result = np.array([float(response[8]), float(response[9]), float(response[10]), float(response[11]), float(response[12]), float(response[13]), float(pos)])
        result = np.round(result, 2)
        return result

    def get_init_pos(self):
        """Get initial end effector position"""
        message = "ReadActPos,0,;"
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success and not self.debug:
            # TODO: handle failure in arm server
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        
        # Check if response contains "ok"
        if b'ok' not in response.lower() and not self.debug:
            print("[ERROR]: Server response does not contain 'ok'")
            print(f"Server response was: {response}")
            sys.exit(1)

        if self.debug:
            # TODO: implement debug mode
            return np.zeros(27)

        response = response.decode().split(',')
        self.init_pos = {}
        self.init_pos['dJ1'] = float(response[2])
        self.init_pos['dJ2'] = float(response[3])
        self.init_pos['dJ3'] = float(response[4])
        self.init_pos['dJ4'] = float(response[5])
        self.init_pos['dJ5'] = float(response[6])
        self.init_pos['dJ6'] = float(response[7])

        self.init_pos['dX'] = float(response[8])
        self.init_pos['dY'] = float(response[9])
        self.init_pos['dZ'] = float(response[10])

        self.init_pos['dRx'] = float(response[11])
        self.init_pos['dRy'] = float(response[12])
        self.init_pos['dRz'] = float(response[13])

        self.init_pos['dTcp_X'] = float(response[14])
        self.init_pos['dTcp_Y'] = float(response[15])
        self.init_pos['dTcp_Z'] = float(response[16])

        self.init_pos['dTcp_Rx'] = float(response[17])
        self.init_pos['dTcp_Ry'] = float(response[18])
        self.init_pos['dTcp_Rz'] = float(response[19])

        self.init_pos['dUcs_X'] = float(response[20])
        self.init_pos['dUcs_Y'] = float(response[21])
        self.init_pos['dUcs_Z'] = float(response[22])

        self.init_pos['dUcs_Rx'] = float(response[23])
        self.init_pos['dUcs_Ry'] = float(response[24])
        self.init_pos['dUcs_Rz'] = float(response[25])

        self.init_pos['gripper'] = self.gripper.get_position()

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return False
        
    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        quat_xyzw = obs[3:]
        quat_wxyz = np.concatenate([quat_xyzw[-1:], quat_xyzw[:-1]])
        obs[3:] = quat_wxyz
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        # rpy = action[3:7]
        # deg2rad = np.pi / 180
        # rpy = rpy * deg2rad
        # quat_xyzw = np.array(quaternion_from_euler(rpy[0], rpy[1], rpy[2]))
        # action[3:7] = quat_xyzw
        return action
           
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'tracker'):
                if hasattr(self.tracker, 'is_running') and self.tracker.is_running:
                    self.tracker.stop_tracking()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def process_obs(self, obs):
        """
        Process the observation and calculate the reward.
        """
        # TODO: implement reward function
        reward = 0
        max_reward = 0
        for target in self.targets:
            target_pos = self.get_3d_obs_by_name(target)[0]
            reward =  -np.linalg.norm(target_pos - obs[:3])
            max_reward = max(max_reward, reward)

        # if self.task == 'move_to_pose':
        #     reward =  -np.linalg.norm(self.target.get_position() -
        #                        obs[:3])
        # elif self.task == 'lift_lid':
        #     grip_to_block = -np.linalg.norm(
        #         self._block.get_position() - obs[:3])
        #     block_to_target = -np.linalg.norm(
        #         self._block.get_position() - self._target.get_position())
        #     reward =  grip_to_block + block_to_target
        # elif self.task == 'grasp_lid':
        #     grasp_lid_reward = -np.linalg.norm(
        #         self.lid.get_position() - obs[:3])
        #     lift_lid_reward = -np.linalg.norm(
        #         self.lid.get_position() - self.success_detector.get_position())
        #     reward =  grasp_lid_reward + lift_lid_reward

        if reward > 0:
            terminate = True
        else:
            terminate = False
        return obs, reward, terminate

