from bvh_utils import *
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    
    #---------------你的代码------------------#
    vertex_num = vertex_translation.shape[0]
    for i in range(vertex_num):
        vertex_trans = np.zeros(vertex_translation.shape[1])
        for j in range(4):
            if skinning_weight[i, j] <= 1e-5:
                continue

            # 当前顶点对应的骨骼
            skin_id = skinning_idx[i, j]
            # Bind Pose下的局部坐标
            local_r = T_pose_vertex_translation[i] - T_pose_joint_translation[skin_id]
            # 当前位置
            cur_r = R.from_quat(joint_orientation[skin_id]).apply(local_r) + joint_translation[skin_id]

            # LBS加权平均
            vertex_trans += skinning_weight[i, j] * cur_r

        vertex_translation[i] = vertex_trans

    return vertex_translation