import numpy as np
from scipy.spatial.transform import Rotation as R


def get_nor(v):
    return v/np.linalg.norm(v)


def calc_angle_between_cur_end(chain_offset, chain_pos, chain_ori,
                               target_pose, end_joint, cur_joint):
    """
    CCD计算目标点与End目标点的角度
    """

    cur_position = chain_pos[cur_joint]
    end_position = chain_pos[end_joint]
    # current joint to target vector with end joint to target vector
    cur_to_end = get_nor(end_position - cur_position)
    cur_to_target = get_nor(target_pose - cur_position)

    rotation_radius = np.arccos(np.clip(np.dot(cur_to_end, cur_to_target), -1, 1))
    rotation_axis = get_nor(np.cross(cur_to_end, cur_to_target))
    rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)
    chain_ori[cur_joint] = rotation_vector * chain_ori[cur_joint]

    chain_rot = [chain_ori[i] if i == 0 else (R.inv(chain_ori[i - 1])) * chain_ori[i]
                 for i in range(len(chain_ori))]

    for i in range(cur_joint + 1, end_joint + 1):
        if i < end_joint + 1:
            chain_ori[i] = chain_ori[i - 1] * chain_rot[i]
        chain_pos[i] = chain_pos[i - 1] + chain_ori[i - 1].apply(chain_offset[i])

    return chain_pos, chain_ori


def fk_update_joints(meta_data, chain_orientations, chain_positions, joint_positions, joint_orientations):
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    joint_rotations = R.identity(len(meta_data.joint_name))
    for k, v in enumerate(meta_data.joint_parent):
        if v != -1:
            joint_rotations[k] = R.inv(R.from_quat(joint_orientations[v])) * R.from_quat(joint_orientations[k])
        else:
            joint_rotations[k] = R.from_quat(joint_orientations[k])

    for i in range(len(path2) - 1):
        joint_orientations[path2[i + 1]] = chain_orientations[i].as_quat()
    joint_orientations[path2[-1]] = chain_orientations[len(path2) - 1].as_quat()
    for i in range(len(path1) - 1):
        joint_orientations[path1[~i]] = chain_orientations[i + len(path2)].as_quat()

    for j in range(len(path)):
        joint_positions[path[j]] = chain_positions[j]

    for i, p in enumerate(meta_data.joint_parent):
        if p == -1:
            continue
        if meta_data.joint_name[i] not in path_name:
            joint_orientations[i] = (R.from_quat(joint_orientations[p]) * joint_rotations[i]).as_quat()
            joint_positions[i] = joint_positions[p] + R.from_quat(joint_orientations[p]).apply(
                meta_data.joint_initial_position[i] - meta_data.joint_initial_position[p])


def construct_chains(meta_data, joint_positions, joint_orientations):
    """
    按是否通过Root节点构建正向FK链
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    chain_orient = []
    chain_pos = [joint_positions[node] for node in path]
    chain_offset = [np.array([0., 0., 0.]) if i == 0 else meta_data.joint_initial_position[path[i]]
                    - meta_data.joint_initial_position[path[i - 1]] for i in range(len(path))]

    for i in range(len(path2) - 1):
        chain_orient.append(R.from_quat(joint_orientations[path2[i + 1]]))
    chain_orient.append(R.from_quat(joint_orientations[path2[-1]]))
    for i in range(len(path1) - 1):
        chain_orient.append(R.from_quat(joint_orientations[path1[~i]]))
    chain_orient.append(R.identity())

    return chain_offset, chain_pos, chain_orient

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

    chain_offsets, chain_positions, chain_orientations = \
        construct_chains(meta_data, joint_positions, joint_orientations)
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    end_joint = meta_data.end_joint
    end_index = path_name.index(end_joint)

    k = 0
    while np.linalg.norm(joint_positions[path[end_index]] - target_pose) >= 1e-2 and k <= 20:
        for i in range(end_index - 1, -1, -1):
            print(path_name[i])
            cur_index = i
            calc_angle_between_cur_end(chain_offsets, chain_positions, chain_orientations,
                                       target_pose, end_index, cur_index)
        k += 1
        print(k)

    fk_update_joints(meta_data, chain_orientations, chain_positions, joint_positions, joint_orientations)

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