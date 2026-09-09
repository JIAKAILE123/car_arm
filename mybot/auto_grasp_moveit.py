#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time
import math

# ROS 消息和服务
from gazebo_msgs.msg import ModelStates
from linkattacher_msgs.srv import AttachLink
from geometry_msgs.msg import Pose, Point

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class AutoGrasperWithMoveit(Node):
    def __init__(self):
        super().__init__('auto_grasper_with_moveit_node')

        # --- MoveIt! 初始化 ---
        self.get_logger().info("MoveItPy 初始化完成")
        
        # 获取规划实体

        # --- ROS 服务和话题客户端/订阅者 ---
        self.model_states_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states', self.model_states_callback, 10
        )
        self.attach_cli = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_cli = self.create_client(AttachLink, '/DETACHLINK')
        
        # 等待服务
        while not self.attach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ATTACHLINK 服务不可用, 正在等待...')
        while not self.detach_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DETACHLINK 服务不可用, 正在等待...')

        self.get_logger().info('自动抓取节点已准备就绪.')
        
        # --- 内部变量 ---
        self.latest_model_states = None
        self.robot_name = 'six_arm'
        self.hand_link_name = 'grasping_frame' # 使用 SRDF 中定义的末端执行器父连杆

    def model_states_callback(self, msg):
        self.latest_model_states = msg


    def find_and_pick_closest_cube(self):
        """完整的寻找、移动、抓取、释放流程"""
        if self.latest_model_states is None:
            self.get_logger().warn('尚未收到任何模型状态信息，请稍等...')
            return

        try:

            # --- 2. 感知与决策 ---
            robot_pose = self.latest_model_states.pose[self.latest_model_states.name.index(self.robot_name)]
            
            cubes = [{'name': name, 'pose': self.latest_model_states.pose[i]} 
                     for i, name in enumerate(self.latest_model_states.name) 
                     if name.startswith('red_cube')]
            
            if not cubes:
                self.get_logger().warn('在场景中没有找到任何 "red_cube"!')
                return
            
            closest_cube = min(cubes, key=lambda cube: math.sqrt(
                (cube['pose'].position.x - robot_pose.position.x)**2 +
                (cube['pose'].position.y - robot_pose.position.y)**2
            ))
            
            target_cube_name = closest_cube['name']
            target_cube_pose = closest_cube['pose']
            self.get_logger().info(f"决策完成: 最近的方块是 '{target_cube_name}'.")

            # --- 3. 执行抓取流程 ---
            # a. 移动到物体上方 (预抓取)
            pre_grasp_pose = create_pose(
                target_cube_pose.position.x, 
                target_cube_pose.position.y, 
                target_cube_pose.position.z + 0.10, # 在物体上方10cm
                1.0, 0.0, 0.0, 0.0 # 面向下的姿态
            )
            self.move_to_pose(pre_grasp_pose)

            
            # d. 调用服务进行吸附
            self.get_logger().info("调用服务进行吸附...")
            self.call_attach_service(self.robot_name, self.hand_link_name, target_cube_name, 'link')
            time.sleep(10) # 等待吸附生效


            self.get_logger().info("调用服务进行释放...")
            self.call_detach_service(self.robot_name, self.hand_link_name, target_cube_name, 'link')
            
            self.get_logger().info("任务完成!")

        except ValueError as e:
            self.get_logger().error(f"处理模型状态时出错: {e}. 您的机器人名 '{self.robot_name}' 是否在Gazebo中?")
        except Exception as e:
            self.get_logger().error(f"发生未知错误: {e}")

    def call_attach_service(self, model1, link1, model2, link2):
        req = AttachLink.Request()
        req.model1_name = model1; req.link1_name = link1
        req.model2_name = model2; req.link2_name = link2
        future = self.attach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"抓取结果: {future.result().message}")
        else:
            self.get_logger().error('调用抓取服务时发生异常')

    def call_detach_service(self, model1, link1, model2, link2):
        req = AttachLink.Request()
        req.model1_name = model1; req.link1_name = link1
        req.model2_name = model2; req.link2_name = link2
        future = self.detach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"释放结果: {future.result().message}")
        else:
            self.get_logger().error('调用释放服务时发生异常')


def main(args=None):
    rclpy.init(args=args)
    
    grasper_node = AutoGrasperWithMoveit(node_name="auto_grasper_with_moveit")
    
    # 需要在新线程中运行spin，以防阻塞主流程
    import threading
    thread = threading.Thread(target=rclpy.spin, args=(grasper_node,), daemon=True)
    thread.start()

    time.sleep(3) # 等待MoveIt和模型状态信息初始化
    
    grasper_node.find_and_pick_closest_cube()
    
    rclpy.shutdown()
    thread.join()

if __name__ == '__main__':
    main()
