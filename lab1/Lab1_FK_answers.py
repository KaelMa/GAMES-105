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


def recur_cal_joint(joint_lines, joint_name, joint_parent, joint_offsets, parent_index):
    names = joint_lines[0].split()
    is_end = False
    if names[0].startswith('JOINT'):
        name = names[1]
    else:
        name = joint_name[parent_index] + '_end'
        is_end = True

    joint_name.append(name)
    joint_parent.append(parent_index)

    offsets = joint_lines[2].split()
    joint_offsets.extend([float(x) for x in offsets[1:]])

    parent_index = len(joint_name) - 1

    if not is_end:
        lines = joint_lines[4:]
        recur_cal_joint(lines, joint_name, joint_parent, joint_offsets, parent_index)

def load_hierarchy_data(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_offsets = []

    root_pos = 0
    joint_pos_list = []
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                root_pos = i
            elif lines[i].startswith('    JOINT'):
                joint_pos_list.append(i)
            elif lines[i].startswith('}'):
                joint_pos_list.append(i - 1)

        root_name = lines[root_pos].split()[1]
        joint_name.append(root_name)
        joint_parent.append(-1)
        offsets = lines[root_pos + 2].split()
        joint_offsets.extend([float(x) for x in offsets[1:]])

        for j in range(0, len(joint_pos_list) - 1, 1):
            start = joint_pos_list[j]
            end = joint_pos_list[j + 1] - 1
            joint_lines = lines[start:end]
            parent_index = 0

            # recursive calc joints
            recur_cal_joint(joint_lines, joint_name, joint_parent, joint_offsets, parent_index)

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
    joint_positions = None
    joint_orientations = None
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