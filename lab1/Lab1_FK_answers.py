import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def recur_cal_joint(joint_lines, joint_name, joint_parent, joint_offsets, parent_index, indent_num):
    names = joint_lines[0].split()
    is_end = False
    if names[0].startswith('JOINT') or names[0].startswith('ROOT'):
        name = names[1]
    else:
        name = joint_name[parent_index] + '_end'
        is_end = True

    # name
    joint_name.append(name)
    # offset
    offsets = joint_lines[2].split()
    joint_offsets.extend([float(x) for x in offsets[1:]])
    # parent index
    joint_parent.append(parent_index)
    parent_index = len(joint_name) - 1
    # find child problem
    if not is_end:
        joint_start_nums = []
        for i in range(len(joint_lines)):
            if joint_lines[i].find('JOINT') == indent_num or \
                    joint_lines[i].find('End') == indent_num:
                joint_start_nums.append(i)
        joint_start_nums.append(len(joint_lines))
        indent_num += 4

        for i in range(len(joint_start_nums) - 1):
            start = joint_start_nums[i]
            end = joint_start_nums[i + 1]
            lines = joint_lines[start: end]
            recur_cal_joint(lines, joint_name, joint_parent, joint_offsets, parent_index, indent_num)


def load_hierarchy_data(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_offsets = []

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                root_pos = i
            elif lines[i].startswith('}'):
                end_pos = i + 1
                break

        joint_lines = lines[root_pos:end_pos]
        recur_cal_joint(joint_lines, joint_name, joint_parent, joint_offsets, -1, 4)

    joint_offset = np.array(joint_offsets).reshape(-1, 3)
    return joint_name, joint_parent, joint_offset



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """

    joint_name = None
    joint_parent = None
    joint_offset = None

    joint_name, joint_parent, joint_offset = load_hierarchy_data(bvh_file_path)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """

    length = len(joint_name)
    init_value = R.identity(length).as_quat()
    joint_orientations = np.asarray(init_value, dtype=np.float64)
    joint_positions = np.empty((length, 3), dtype=np.float64)

    # orientation
    for i in range(length):
        k = 0
        if joint_name[i].endswith('_end'):
            r = R.identity()
        else:
            start = 3 + k * 3
            data = motion_data[frame_id, start: start+3]
            r = R.from_euler('xyz', data, degrees=True)
            k += 1

        p_index = max(joint_parent[i], 0)
        p_r = R.from_quat(joint_orientations[p_index])
        q = p_r * r

        joint_orientations[i] = q.as_quat()

    # position
    for j in range(length):
        if j == 0:
            data = motion_data[frame_id, 0: 3]
            offset = joint_offset[j]
            joint_positions[j] = data + offset
        else:
            offset = joint_offset[j]
            p_index = joint_parent[j]
            p_position = joint_positions[p_index]

            q = R.from_quat(joint_orientations[j]).apply(offset)
            position = p_position + q
            joint_positions[j] = position

    return joint_positions, joint_orientations

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    motion_data = None
    return motion_data