import numpy as np
from scipy.spatial.transform import Rotation as R


def get_nor(v):
    return v/np.linalg.norm(v)


def calc_angle_between_cur_end(joint_positions, joint_orientations, target_pose, path, end_index, cur_index):
    """
    CCD计算目标点与End目标点的角度
    """
    end_joint = path[end_index]
    cur_joint = path[cur_index]

    cur_position = joint_positions[cur_joint]
    end_position = joint_positions[end_joint]
    # current joint to target vector with end joint to target vector
    cur_to_end = get_nor(end_position - cur_position)
    cur_to_target = get_nor(target_pose - cur_position)

    rotation_radius = np.arccos(np.dot(cur_to_end, cur_to_target))
    rotation_axis = get_nor(np.cross(cur_to_end, cur_to_target))
    rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)

    for i in range(cur_index, end_index):
        ic = path[i]
        joint_orientations[ic] = (rotation_vector * R.from_quat(joint_orientations[ic])).as_quat()

        icc = path[i + 1]
        joint_positions[icc] = joint_positions[cur_joint] + \
                              rotation_vector.apply(joint_positions[icc] - joint_positions[cur_joint])

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    end_joint = meta_data.end_joint
    end_index = path_name.index(end_joint)

    k = 0
    while np.linalg.norm(joint_positions[path[end_index]] - target_pose) >= 1e-2 and k <= 100:
        for i in range(end_index - 1, 0, -1):
            # from 6 to 1
            cur_index = i
            calc_angle_between_cur_end(joint_positions, joint_orientations, target_pose, path, end_index, cur_index)
        k += 1
        print(k)
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations