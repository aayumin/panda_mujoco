"""Demonstrates the Franka Emika Robot System model for MuJoCo."""

import time
import threading
from threading import Thread

from glob import glob
import os
os.environ["MUJOCO_GL"] = "glfw"  # or "osmesa" if needed

import glfw
import mujoco
import numpy as np

start_flag = False
lock = threading.Lock()

class Demo:

    # qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    qpos0 = [0, 0, 0, -0.785, 0, 1.571, 0.785]
    
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 480, 640  # Rendering window resolution.
    fps = 60  # Rendering framerate.

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        # self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        
        
        
        self.id_to_body_name = {}
        self.name_to_body_id = {}
        for b_idx in range(self.model.nbody):
            self.id_to_body_name[b_idx] = self.model.body(b_idx).name
            self.name_to_body_id[self.model.body(b_idx).name] = b_idx

    def gripper(self, open=True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
        self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
                f"panda_joint{i}"
            ).qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (
                J[:, dofadr].T @ np.diag(self.K) @ error
            )
            self.data.actuator(f"panda_joint{i}").ctrl -= (
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )
    
      
    def check_collision(self, data, obj_name1 = None, obj_name2 = None):
        with lock:
            num_contacts = data.ncon
            # print(f"\nTime: {round(data.time, 3)}, # of contacts: {data.ncon}")
            for i in range(num_contacts):
                ctt = data.contact[i]
                ctt_obj_1 = data.geom(ctt.geom1).name
                ctt_obj_2 = data.geom(ctt.geom2).name
                
                contact_pt = [round(v,2) for v in ctt.pos]
                contact_normal = [round(v,2) for v in ctt.frame[:3]]
                
                if obj_name1 is not None and obj_name2 is None and obj_name1 in [ctt_obj_1, ctt_obj_2]:
                    # print(f"\t\tcontact => ({ctt_obj_1}, {ctt_obj_2})  : ", contact_pt, contact_normal)
                    return contact_pt, contact_normal
                
                if obj_name1 is not None and obj_name2 is not None and obj_name1 in [ctt_obj_1, ctt_obj_2] and obj_name2 in [ctt_obj_1, ctt_obj_2]:
                    # print(f"\t\tcontact => ({ctt_obj_1}, {ctt_obj_2})  : ", contact_pt, contact_normal)
                    return contact_pt, contact_normal
                
            return None

    def add_noise_to_friction(self, body_name, noise_std=0.1):
        cur_geom = self.model.geom(body_name)
        
        original_friction = cur_geom.friction
        noisy_friction = original_friction + np.random.normal(0, noise_std, size=original_friction.shape)
        print(f"original_friction: {original_friction}, noisy_friction: {noisy_friction}")
        cur_geom.friction = noisy_friction
        
    def add_noise_to_hand_motion(self, data, noise_std = 0.2): ## data = [X, X, X, ...] : free length list
        data_arr = np.array(data)
        noisy_data = data_arr + np.random.normal(0, noise_std, size=data_arr.shape)
        return noisy_data.tolist()
        
        
        
        
    def pick_sample_name(self):
        files = os.listdir("./dataset")
        recent_num = int(sorted(files)[-1].split("\\")[-1].split(".")[0].split("_")[0])
        
        return str(recent_num+1).zfill(4)
        
 
    def hit(self):
        
        
        sample_name = self.pick_sample_name()
        # self.add_noise_to_friction("cup")
        self.add_noise_to_friction("cup", 0.5)
        
        
        xpos0 = self.data.body("panda_hand").xpos.copy()
        xpos_d = xpos0
        xquat0 = self.data.body("panda_hand").xquat.copy()
        # self.gripper(False)
        
        # input("press any key")
        # global start_flag
        # start_flag = True
        
        initial_collision_flag = False
        
        x1, x2, x3 = 0.43, 0.7, 0.9
        y1, y2, y3 = 0, 0, 1.2
        z1, z2, z3 = 0.96, -0.8, -1
        x1, x2, x3, y1, y2, y3, z1, z2, z3 = self.add_noise_to_hand_motion(data=[x1, x2, x3, y1, y2, y3, z1, z2, z3])
        target_x = list(np.linspace(x2, x1, 2000))
        target_y = list(np.linspace(y2, y1, 2000))
        target_z = list(np.linspace(z2, z1, 2000))
        
        while self.run:
            if len(target_x) == 0:
                break
            xpos_d = xpos0 + [target_x.pop(), target_y.pop(), target_z.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            # self.check_collision(self.data, "cup", "floor")
            result = self.check_collision(self.data, "cup", "panda_hand")
            if result is not None:
                initial_collision_flag = True
                self.prepare_record(sample_name, result[0], result[1])
                
        
        
        
        target_x = list(np.linspace(x3, x2, 500))
        target_y = list(np.linspace(y3, y2, 500))
        target_z = list(np.linspace(z3, z2, 500))
        while self.run:
            if len(target_x) == 0:
                break
            xpos_d = xpos0 + [target_x.pop(), target_y.pop(), target_z.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            
            
            
            
            result = self.check_collision(self.data, "cup", "panda_hand")
            if initial_collision_flag == False and result is not None:
                initial_collision_flag = True
                self.prepare_record(sample_name, result[0], result[1])
            if initial_collision_flag and result is not None:
                self.record_contact_point(sample_name, "hand", result[0], result[1])
            
            
            result = self.check_collision(self.data, "cup", "floor")
            if initial_collision_flag and result is not None:
                self.record_contact_point(sample_name, "floor", result[0], result[1])
                
            
            # time.sleep(1e-6)
            
        for _ in range(1000):
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            # result = self.check_collision(self.data, "cup", "floor")
            # if result is not None: break
        

    def prepare_record(self, sample_name, init_pt, init_normal):
        print(f"start record with initial point:  {init_pt}, {init_normal}")
        for obj_name in ["hand", "floor"]:
            fname = sample_name + f"_{obj_name}"
            with open(f"dataset/{fname}.csv", "w") as f:
                f.write(",".join(map(str, init_pt)))
                f.write(",")
                f.write(",".join(map(str, init_normal)))
                f.write("\n")
        
        
    # def check_hit_point(self, sample_name, obj_name, contact_pt, contact_normal):
    def record_contact_point(self, sample_name, obj_name, contact_pt, contact_normal):
        # input("press any key")
        
        # max_iter = 1e8
        # while max_iter:
        #     max_iter -= 1
        #     result = self.check_collision(self.data, "cup", "panda_hand")
        #     if result is not None: break
        
        # if result is None: return
        
        fname = sample_name + f"_{obj_name}"
        # contacts_with_hand = []
        # contacts_with_floor = []
        
        
        # sample_len = 100000 
        # for _ in range(sample_len):
        #     mujoco.mj_step(self.model, self.data)
        #     result1 = self.check_collision(self.data, "cup", "panda_hand")
        #     result2 = self.check_collision(self.data, "cup", "floor")
        #     if result1 is not None: contacts_with_hand.append(result1)
        #     if result2 is not None: contacts_with_floor.append(result1)
            
            
        
        # print(f"len_hand: {len(contacts_with_hand)}, len_floor: {len(contacts_with_floor)}")
        with open(f"dataset/{fname}.csv", "a") as f:
            f.write(",".join(map(str, contact_pt)))
            f.write(",")
            f.write(",".join(map(str, contact_normal)))
            f.write("\n")
            

    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.hit)
        step_thread.start()
        
        
        # check_thread = Thread(target=self.check_hit_point)
        # check_thread.start()
        
        self.render()
        
        


if __name__ == "__main__":
    Demo().start()
