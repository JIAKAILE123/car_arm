# /gripper_controller/follow_joint_trajectory

ros2 action send_goal /gripper_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [finger_joint1],
    points: [
      { positions: [0.0], time_from_start: { sec: 0, nanosec: 200 } },
      { positions: [0.1], time_from_start: { sec: 2, nanosec: 200 } },
      { positions: [0.2], time_from_start: { sec: 4, nanosec: 200 } }
    ]
  }
}"
