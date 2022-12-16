# 以下部分均为可更改部分

from answer_task1 import *

class CharacterController():
    def __init__(self, controller) -> None:
        # self.motions = [BVHMotion('motion_material/kinematic_motion/long_run.bvh'),
        #                 BVHMotion('motion_material/kinematic_motion/long_run_mirror.bvh'),
        #                 BVHMotion('motion_material/kinematic_motion/long_walk.bvh'),
        #                 BVHMotion('motion_material/kinematic_motion/long_walk_mirror.bvh')]

        self.motions = [BVHMotion('motion_material/idle.bvh'),
                        BVHMotion('motion_material/run_forward.bvh'),
                        BVHMotion('motion_material/walk_and_ture_right.bvh'),
                        BVHMotion('motion_material/walk_and_turn_left.bvh'),
                        BVHMotion('motion_material/walk_forward.bvh'),
                        BVHMotion('motion_material/walkF.bvh')]
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_pose_feature = None
        self.cur_frame = 0
        self.cur_motion = 0
        self.dt = 1/60
        self.db = None
        self.init_db()

    def init_db(self):
        """
        初始化feature vector cost db
        """
        motions_data = []
        for i in range(len(self.motions)):
            motion_data = []
            cur_motion_position = self.motions[i].joint_position
            cur_motion_rotation = self.motions[i].joint_rotation
            frame_num = cur_motion_position.shape[0]
            for j in range(frame_num - 1):
                # root velocity, left/right joint local position and velocity
                data = []
                feature = []
                root_vel = (cur_motion_position[j + 1][0] - cur_motion_position[j][0]) / self.dt
                feature.append(root_vel)

                l_foot_index = self.motions[i].joint_name.index('lToeJoint_end')
                l_foot_pos = cur_motion_position[j][l_foot_index] - cur_motion_position[j][0]
                l_foot_vel = ((cur_motion_position[j + 1][l_foot_index] - cur_motion_position[j + 1][0])
                              - l_foot_pos) / self.dt
                feature.append(l_foot_pos)
                feature.append(l_foot_vel)

                r_foot_index = self.motions[i].joint_name.index('rToeJoint_end')
                r_foot_pos = cur_motion_position[j][r_foot_index] - cur_motion_position[j][0]
                r_foot_vel = ((cur_motion_position[j + 1][r_foot_index] - cur_motion_position[j + 1][0])
                              - r_foot_pos) / self.dt
                feature.append(r_foot_pos)
                feature.append(r_foot_vel)
                data.append(feature)

                # trajectory in 20 40 60 80 100 frame
                traj = []
                t1 = cur_motion_position[min(j + 20, frame_num - 1)][0] - cur_motion_position[j][0]
                t2 = cur_motion_position[min(j + 40, frame_num - 1)][0] - cur_motion_position[j][0]
                t3 = cur_motion_position[min(j + 60, frame_num - 1)][0] - cur_motion_position[j][0]
                t4 = cur_motion_position[min(j + 80, frame_num - 1)][0] - cur_motion_position[j][0]
                t5 = cur_motion_position[min(j + 100, frame_num - 1)][0] - cur_motion_position[j][0]
                traj.append(t1)
                traj.append(t2)
                traj.append(t3)
                traj.append(t4)
                traj.append(t5)

                root_rot_inv = R.from_quat(cur_motion_rotation[j][0]).inv()
                r1 = R.from_quat(cur_motion_rotation[min(j + 20, frame_num - 1)][0]) * root_rot_inv
                r2 = R.from_quat(cur_motion_rotation[min(j + 40, frame_num - 1)][0]) * root_rot_inv
                r3 = R.from_quat(cur_motion_rotation[min(j + 60, frame_num - 1)][0]) * root_rot_inv
                r4 = R.from_quat(cur_motion_rotation[min(j + 80, frame_num - 1)][0]) * root_rot_inv
                r5 = R.from_quat(cur_motion_rotation[min(j + 100, frame_num - 1)][0]) * root_rot_inv
                traj.append(r1)
                traj.append(r2)
                traj.append(r3)
                traj.append(r4)
                traj.append(r5)

                data.append(traj)
                motion_data.append(data)
            motions_data.append(motion_data)

        self.db = motions_data
        print("init feature cost db done")

    def compute_future_cost(self, feature):
        min_cost = 1e5
        best_frame = None

        for i in range(len(self.db)):
            for j in range(len(self.db[i])):
                c0 = np.linalg.norm(feature[0] - self.db[i][j][1][0])
                c1 = np.linalg.norm(feature[1] - self.db[i][j][1][1])
                c2 = np.linalg.norm(feature[2] - self.db[i][j][1][2])
                c3 = np.linalg.norm(feature[3] - self.db[i][j][1][3])
                c4 = np.linalg.norm(feature[4] - self.db[i][j][1][4])

                c5 = np.linalg.norm((feature[5] * self.db[i][j][1][5].inv()).as_rotvec())
                c6 = np.linalg.norm((feature[6] * self.db[i][j][1][6].inv()).as_rotvec())
                c7 = np.linalg.norm((feature[7] * self.db[i][j][1][7].inv()).as_rotvec())
                c8 = np.linalg.norm((feature[8] * self.db[i][j][1][8].inv()).as_rotvec())
                c9 = np.linalg.norm((feature[9] * self.db[i][j][1][9].inv()).as_rotvec())
                t = sum([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9])
                if t <= min_cost:
                    min_cost = t
                    best_frame = (i, j)
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
        self.cur_pose_feature = self.db[0][0]
        feature = None
        t1 = desired_pos_list[1] - desired_pos_list[0]
        t2 = desired_pos_list[2] - desired_pos_list[0]
        t3 = desired_pos_list[3] - desired_pos_list[0]
        t4 = desired_pos_list[4] - desired_pos_list[0]
        t5 = desired_pos_list[5] - desired_pos_list[0]

        r0 = R.from_quat(desired_rot_list[0]).inv()
        r1 = R.from_quat(desired_rot_list[1]) * r0
        r2 = R.from_quat(desired_rot_list[2]) * r0
        r3 = R.from_quat(desired_rot_list[3]) * r0
        r4 = R.from_quat(desired_rot_list[4]) * r0
        r5 = R.from_quat(desired_rot_list[5]) * r0
        feature = [t1, t2, t3, t4, t5, r1, r2, r3, r4, r5]

        best_frame = self.compute_future_cost(feature)
        if self.cur_motion == best_frame[0] and self.cur_frame == best_frame[1]:
            self.cur_motion = (self.cur_motion + 1) % len(self.motions)
            self.cur_frame = (self.cur_frame + 1) % self.motions[self.cur_motion].motion_length
        else:
            self.cur_motion = best_frame[0]
            self.cur_frame = best_frame[1]

        print("best frame: ", best_frame)

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