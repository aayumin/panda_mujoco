"""Demonstrates the Franka Emika Robot System model for MuJoCo."""

import time
from threading import Thread

import os
os.environ["MUJOCO_GL"] = "glfw"  # or "osmesa" if needed

import glfw
import mujoco
import numpy as np


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
        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        
        
        
        self.id_to_body_name = {}
        for b_idx in range(self.model.nbody):
            self.id_to_body_name[b_idx] = self.model.body(b_idx).name

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
        num_contacts = data.ncon
        # print(f"\nTime: {round(data.time, 3)}, # of contacts: {data.ncon}")
        for i in range(num_contacts):
            ctt = data.contact[i]
            ctt_obj_1 = self.id_to_body_name[ctt.geom1]
            ctt_obj_2 = self.id_to_body_name[ctt.geom2]
            
            
            if obj_name1 is not None and obj_name2 is None and obj_name1 in [ctt_obj_1, ctt_obj_2]:
                print(f"\t\tcontact => ({ctt_obj_1}, {ctt_obj_2})  : ", [round(v,2) for v in ctt.pos], [round(v,2) for v in ctt.frame[:3]])
                return True
            
            if obj_name1 is not None and obj_name2 is not None and obj_name1 in [ctt_obj_1, ctt_obj_2] and obj_name2 in [ctt_obj_1, ctt_obj_2]:
                print(f"\t\tcontact => ({ctt_obj_1}, {ctt_obj_2})  : ", [round(v,2) for v in ctt.pos], [round(v,2) for v in ctt.frame[:3]])
                return True
            
        return False


 
    def hit(self):
        xpos0 = self.data.body("panda_hand").xpos.copy()
        xpos_d = xpos0
        xquat0 = self.data.body("panda_hand").xquat.copy()
        self.gripper(False)
        
        target_x = list(np.linspace(0.7, 0.43, 2000))
        target_y = list(np.linspace(0, 0, 2000))
        target_z = list(np.linspace(-1, 0.96, 2000))
        while self.run:
            if len(target_x) == 0:
                break
            xpos_d = xpos0 + [target_x.pop(), target_y.pop(), target_z.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            self.check_collision(self.data, "cup")
        
        
        target_x = list(np.linspace(0.9, 0.7, 500))
        target_y = list(np.linspace(1.2, 0, 500))
        target_z = list(np.linspace(-1, -1, 500))
        while self.run:
            if len(target_x) == 0:
                break
            xpos_d = xpos0 + [target_x.pop(), target_y.pop(), target_z.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            # self.check_collision(self.data, "world")
            self.check_collision(self.data, "cup")
            # time.sleep(1e-6)
            
        for _ in range(10000):
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
        

    def step(self) -> None:
        xpos0 = self.data.body("panda_hand").xpos.copy()
        xpos_d = xpos0
        xquat0 = self.data.body("panda_hand").xquat.copy()
        down = list(np.linspace(-0.45, 0, 2000))
        up = list(np.linspace(0, -0.45, 2000))
        state = "down"
        while self.run:
            if state == "down":
                if len(down):
                    xpos_d = xpos0 + [0, 0, down.pop()]
                else:
                    state = "grasp"
            elif state == "grasp":
                self.gripper(False)
                state = "up"
            elif state == "up":
                if len(up):
                    xpos_d = xpos0 + [0, 0, up.pop()]
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            # time.sleep(1e-3)

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
        self.render()


if __name__ == "__main__":
    Demo().start()
