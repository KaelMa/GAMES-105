# 以下部分均为可更改部分

from answer_task1 import *
from scipy.spatial import KDTree

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = [BVHMotion('motion_material/idle.bvh'),
                        BVHMotion('motion_material/run_forward.bvh'),
                        # BVHMotion('motion_material/walk_and_ture_right.bvh'),
                        # BVHMotion('motion_material/walk_and_turn_left.bvh'),
                        # BVHMotion('motion_material/walk_forward.bvh'),
                        # BVHMotion('motion_material/walkF.bvh')
                        ]

        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_pose_feature = None
        self.cur_frame = 0
        self.cur_motion = 0
        self.dt = 1/60
        self.cost_db = None
        self.init_db()

    def init_db(self):
        """
        初始化feature vector cost db
        """
        motions_data = []
        for i in range(len(self.motions)):
            cur_motion_position = self.motions[i].joint_position
            cur_motion_rotation = self.motions[i].joint_rotation
            frame_num = cur_motion_position.shape[0]
            motion_data = np.empty((frame_num, 10))

            for j in range(frame_num - 1):
                # root velocity, left/right joint local position and velocity

                # feature = []
                # root_vel = (cur_motion_position[j + 1][0] - cur_motion_position[j][0]) / self.dt
                # feature.append(root_vel)
                #
                # l_foot_index = self.motions[i].joint_name.index('lToeJoint_end')
                # l_foot_pos = cur_motion_position[j][l_foot_index] - cur_motion_position[j][0]
                # l_foot_vel = ((cur_motion_position[j + 1][l_foot_index] - cur_motion_position[j + 1][0])
                #               - l_foot_pos) / self.dt
                # feature.append(l_foot_pos)
                # feature.append(l_foot_vel)
                #
                # r_foot_index = self.motions[i].joint_name.index('rToeJoint_end')
                # r_foot_pos = cur_motion_position[j][r_foot_index] - cur_motion_position[j][0]
                # r_foot_vel = ((cur_motion_position[j + 1][r_foot_index] - cur_motion_position[j + 1][0])
                #               - r_foot_pos) / self.dt
                # feature.append(r_foot_pos)
                # feature.append(r_foot_vel)
                # data.append(feature)

                # trajectory in 20 40 60 80 100 frame
                motion_data[j, 0] = np.linalg.norm(cur_motion_position[min(j + 20, frame_num - 1), 0, [0, 2]] - cur_motion_position[j, 0, [0, 2]])
                motion_data[j, 1] = np.linalg.norm(cur_motion_position[min(j + 40, frame_num - 1), 0, [0, 2]] - cur_motion_position[j, 0, [0, 2]])
                motion_data[j, 2] = np.linalg.norm(cur_motion_position[min(j + 60, frame_num - 1), 0, [0, 2]] - cur_motion_position[j, 0, [0, 2]])
                motion_data[j, 3] = np.linalg.norm(cur_motion_position[min(j + 80, frame_num - 1), 0, [0, 2]] - cur_motion_position[j, 0, [0, 2]])
                motion_data[j, 4] = np.linalg.norm(cur_motion_position[min(j + 100, frame_num - 1), 0, [0, 2]] - cur_motion_position[j, 0, [0, 2]])

                motion_data[j, 5] = CharacterController.find_diff_y_axis(cur_motion_rotation[min(j + 20, frame_num - 1)][0], cur_motion_rotation[j][0])
                motion_data[j, 6] = CharacterController.find_diff_y_axis(cur_motion_rotation[min(j + 40, frame_num - 1)][0], cur_motion_rotation[j][0])
                motion_data[j, 7] = CharacterController.find_diff_y_axis(cur_motion_rotation[min(j + 60, frame_num - 1)][0], cur_motion_rotation[j][0])
                motion_data[j, 8] = CharacterController.find_diff_y_axis(cur_motion_rotation[min(j + 80, frame_num - 1)][0], cur_motion_rotation[j][0])
                motion_data[j, 9] = CharacterController.find_diff_y_axis(cur_motion_rotation[min(j + 100, frame_num - 1)][0], cur_motion_rotation[j][0])

            tree = KDTree(motion_data, copy_data = True)
            motions_data.append(tree)

        self.cost_db = motions_data
        print("init feature cost db done")

    @staticmethod
    def find_diff_y_axis(rot1, rot2):
        """
        rot1与rot2关于Y轴旋转角度差异
        """
        r1_y, _ = BVHMotion.decompose_rotation_with_yaxis(rot1)
        r2_y, _ = BVHMotion.decompose_rotation_with_yaxis(rot2)

        diff = (r1_y * r2_y.inv()).as_rotvec()
        return np.linalg.norm(diff)

    @staticmethod
    def find_diff_pos_xz(pos1, pos2):
        return np.linalg.norm(pos1[0, 2], pos2[0, 2])

    def compute_future_cost(self, feature):
        min_cost = 1e5
        best_frame = None

        for i in range(len(self.cost_db)):
            tree = self.cost_db[i]
            dist, idx = tree.query(feature, p = 1)
            if dist <= min_cost:
                min_cost = dist
                best_frame = (i, idx)
        return best_frame

    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''

        # solved by motion matching
        # computer feature vector
        t1 = np.linalg.norm(desired_pos_list[1, [0, 2]] - desired_pos_list[0, [0, 2]])
        t2 = np.linalg.norm(desired_pos_list[2, [0, 2]] - desired_pos_list[0, [0, 2]])
        t3 = np.linalg.norm(desired_pos_list[3, [0, 2]] - desired_pos_list[0, [0, 2]])
        t4 = np.linalg.norm(desired_pos_list[4, [0, 2]] - desired_pos_list[0, [0, 2]])
        t5 = np.linalg.norm(desired_pos_list[5, [0, 2]] - desired_pos_list[0, [0, 2]])

        r1 = CharacterController.find_diff_y_axis(desired_rot_list[1], desired_rot_list[0])
        r2 = CharacterController.find_diff_y_axis(desired_rot_list[2], desired_rot_list[0])
        r3 = CharacterController.find_diff_y_axis(desired_rot_list[3], desired_rot_list[0])
        r4 = CharacterController.find_diff_y_axis(desired_rot_list[4], desired_rot_list[0])
        r5 = CharacterController.find_diff_y_axis(desired_rot_list[5], desired_rot_list[0])
        feature = [t1, t2, t3, t4, t5, r1, r2, r3, r4, r5]

        best_frame = self.compute_future_cost(feature)

        if self.cur_motion == best_frame[0] and self.cur_frame == best_frame[1]:
            self.cur_frame = min(self.cur_frame + 1, self.motions[self.cur_motion].motion_length - 1)
        else:
            self.cur_motion = best_frame[0]
            self.cur_frame = best_frame[1]

        print("best frame: {%d, %d}, cur frame: {%d, %d}" % (best_frame[0], best_frame[1], self.cur_motion, self.cur_frame))

        joint_name = self.motions[self.cur_motion].joint_name
        joint_translation, joint_orientation = self.motions[self.cur_motion].batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]

        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]

        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.