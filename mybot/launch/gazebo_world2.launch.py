import os
import re
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
import xacro

def remove_comments(text):
    """移除XML/HTML注释"""
    pattern = r'<!--(.*?)-->'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def generate_launch_description():
    robot_name_in_model = 'six_arm'
    model_pkg_name = 'mybot_description'
    urdf_name = "originbot_with_rgbd_gazebo_arm.xacro"
    #world_name = 'room_aboxa3.world'
    world_name = 'offic_room.world'

    model_pkg_share = FindPackageShare(package=model_pkg_name).find(model_pkg_name)
    pkg_share = FindPackageShare(package='mybot').find('mybot')
    urdf_model_path = os.path.join(model_pkg_share, f'urdf/{urdf_name}')
    world_file_path = os.path.join(model_pkg_share, f'worlds/{world_name}')

    # 启动 Gazebo（带世界文件）
    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world_file_path],
        output='screen'
    )

    # 解析 xacro 并去除注释
    doc = xacro.parse(open(urdf_model_path))
    xacro.process_doc(doc)
    params = {'robot_description': remove_comments(doc.toxml())}

    # robot_state_publisher 节点
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, params, {"publish_frequency": 15.0}],
        output='screen'
    )

    # 在 Gazebo 中生成机器人（通过 robot_description 话题）
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', robot_name_in_model,
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.0', '-Y', '0.0'
        ],
        output='screen'
    )

    # ========== 使用 spawner 加载控制器（替代 ros2 control 命令） ==========
    # 1. 关节状态广播器
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen'
    )

    # 2. 机械臂轨迹控制器
    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '-c', '/controller_manager'],
        output='screen'
    )

    # 3. 夹爪控制器
    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '-c', '/controller_manager'],
        output='screen'
    )

    # ========== 事件顺序（保证控制器按顺序加载） ==========
    # 当机器人生成完成后，加载关节状态广播器
    evt1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=[load_joint_state_broadcaster]
        )
    )
    # 当关节状态广播器加载完成后，加载手臂控制器
    evt2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_arm_controller]
        )
    )
    # 当手臂控制器加载完成后，加载夹爪控制器
    evt3 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_arm_controller,
            on_exit=[load_gripper_controller]
        )
    )

    # robot_localization EKF 节点
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(pkg_share, 'config/ekf.yaml'), {'use_sim_time': True}]
    )

    ld = LaunchDescription()
    ld.add_action(start_gazebo_cmd)
    ld.add_action(node_robot_state_publisher)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(evt1)
    ld.add_action(evt2)
    ld.add_action(evt3)
    ld.add_action(robot_localization_node)

    return ld
