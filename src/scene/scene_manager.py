import time
from orca_gym.scene.orca_gym_scene import OrcaGymScene
from orca_gym.log import OrcaLog

orca_logger = OrcaLog.get_instance()

import numpy as np

from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.scene.orca_gym_scene import Actor, MaterialInfo, LightInfo, OrcaGymScene
from orca_gym.log import OrcaLog
from scene.random_util import choose_random_indices, get_random_qpos, get_random_transform

orca_log = OrcaLog.get_instance()

class SceneManager:
    def __init__(self, grpc_addr: str, config: dict = {}, env: OrcaGymLocalEnv = None, init_env_callback = None):
        self._scene = OrcaGymScene(grpc_addr)
        self._config = config
        self._config_check()
        self._init_env_callback = init_env_callback
        self._random_count = 0
        self._first_spawn_actor = True
        self._light_random_cycle = self._config.get("light", {}).get("random", {}).get("cycle", 20)

        self.env = env
        self.scene_info = {}

    def register_init_env_callback(self, init_env_callback):
        self._init_env_callback = init_env_callback

    def set_env(self, env: OrcaGymLocalEnv):
        self.env = env

    def get_actors_joints_in_env(self)-> list[str]:
        '''
        OrcaStudio导出xml时，会携带命名空间前缀， 这里的前缀就是actor name, 
        因为actor spawnable到场景中，会默认放在全局配置Components下
        '''
        actor_names = self._config.get("actor", {}).get("names", [])
        actor_joints = self._config.get("actor", {}).get("joints", [])
        actor_joints = [actor_names[i] + f"_{actor_joints[i]}" for i in range(len(actor_names))]

        return actor_joints

    def serialize_scene(self, in_scene_actors: list[str])-> dict:
        self.scene_info = {}
        actor_names = self._config.get("actor", {}).get("names", [])
        actor_joints = self.get_actors_joints_in_env()
        for i in range(len(actor_joints)):
            if actor_joints[i] in in_scene_actors:
                self.scene_info[actor_names[i]] = {
                    "joint_name": actor_joints[i],
                    "joint_qpos": list(self.get_actor_qpos(actor_joints[i])),
                }
        return self.scene_info

    def get_scene_info(self)-> dict:
        return self.scene_info

    def get_task_config(self)-> dict:
        task_config = self._config.get("task", {})
        return task_config

    def set_actor_qpos(self, joint_name: str, qpos: np.array):
        '''
        @param:
            joint_name: 关节名称
            qpos: 关节位置
        '''
        self.env.set_joint_qpos({joint_name: qpos})

    def get_actor_qpos(self, joint_name: str):
        qpos = self.env.query_joint_qpos([joint_name])[joint_name]
        return qpos

    def update_actor_qpos(self, restore: bool = False, scene_info: dict = None):
        '''
        @parma
            restore: 是否复原场景
            scene_info: 复原场景时用到的场景信息
        '''
        self.reset_actor_pos()
        in_scene_actors = []

        if restore:
            for actor_name, actor_info in scene_info.items():
                self.set_actor_qpos(actor_info["joint_name"], actor_info["joint_qpos"])
                in_scene_actors.append(actor_info["joint_name"])
        else:
            random_config = self._config.get("actor", {}).get("random", {})
            is_random_qpos = random_config.get("qpos", False)
            if is_random_qpos:
                actor_random_nums = random_config.get("nums", [0, 0])
                pick_nums = (np.random.randint(actor_random_nums[0], actor_random_nums[1]) 
                            if actor_random_nums[0] != actor_random_nums[1] else actor_random_nums[0])
                if pick_nums > 0:
                    # 随机挑选pick_nums索引，从range(len(actor_names))中挑选
                    joints = self.get_actors_joints_in_env()
                    joints_dof = self._config.get("actor", {}).get("joints_dof", [])
                    pick_indices = np.random.choice(range(len(joints)), pick_nums, replace=False)
                    for i in pick_indices:
                        joint_name = joints[i]
                        in_scene_actors.append(joint_name)
                        dof = joints_dof[i]

                        if dof == 6:
                            bound_position = random_config.get("six_dof", {}).get("bound_position")
                            bound_rotation = random_config.get("six_dof", {}).get("bound_rotation", [[0, 0], [0, 0], [0, 0]])             
                            center = random_config.get("six_dof", {}).get("center")
                            bound_position = [[center[0] + bound_position[0][0], center[0] + bound_position[0][1]], 
                                              [center[1] + bound_position[1][0], center[1] + bound_position[1][1]], 
                                              [center[2] + bound_position[2][0], center[2] + bound_position[2][1]]]

                            qpos_bound = np.concatenate([bound_position, bound_rotation])

                        elif dof == 3:
                            bound = random_config.get("three_dof", {}).get("bound")                            
                            qpos_bound = np.concatenate([bound])
                        elif dof == 1:
                            bound = random_config.get("one_dof", {}).get("bound")
                            qpos_bound = np.concatenate([bound])
                        qpos = get_random_qpos(qpos_bound, dof)
                        orca_log.info(f"set actor qpos: {joint_name}, {qpos}")
                        self.set_actor_qpos(joint_name, qpos)
                        self.env.mj_forward()
        
        self.serialize_scene(in_scene_actors)

    def reset_actor_pos(self):
        joints_dof = self._config.get("actor", {}).get("joints_dof", [])
        joint_names = self.get_actors_joints_in_env()
        for i in range(len(joints_dof)):
            dof = joints_dof[i]
            joint_name = joint_names[i]
            if dof == 6:
                self.set_actor_qpos(joint_name, [100000, 100000, 1, 1, 0, 0, 0])

    def spawn_scene(self):
        #将所有的actor加入到场景中
        if self.is_update_light():
            self.publish_scene_without_init_env()
            self.spawn_actors()
            self.spawn_lights()
            self.publish_scene()
            self._first_spawn_actor = False
        if self._first_spawn_actor:
            self.publish_scene_without_init_env()
            self._first_spawn_actor = False
            self.spawn_actors()
            self.publish_scene()
        self._random_count += 1

    def spawn_actors(self):
        actor_names = self._config.get("actor", {}).get("names", [])
        actor_spawnables = self._config.get("actor", {}).get("spawnable", [])
        joints_dof = self._config.get("actor", {}).get("joints_dof", [])
        # 初始位置为无穷远
        for i in range(len(actor_names)):
            dof = joints_dof[i]
            actor_name = actor_names[i]
            actor_spawnable = actor_spawnables[i]
            orca_log.info(f"spawn actor: {actor_name}, {actor_spawnable}")
            if dof == 6:
                self.add_actor(actor_name, actor_spawnable, [100000, 100000, 1], [0, 0, 0, 1])
            elif dof == 3:
                center = self._config.get("actor", {}).get("random", {}).get("three_dof", {}).get("center")
                self.add_actor(actor_name, actor_spawnable, [center[0], center[1], center[2]], [0, 0, 0, 1])
            elif dof == 1:
                center = self._config.get("actor", {}).get("random", {}).get("one_dof", {}).get("center")
                self.add_actor(actor_name, actor_spawnable, [center[0], center[1], center[2]], [0, 0, 0, 1])
    
    def is_update_light(self):
        light_config = self._config.get("light", {})
        light_random = light_config.get("random", {})
        is_random_position = light_random.get("position", False)
        is_random_rotation = light_random.get("rotation", False)
        if not (is_random_position or is_random_rotation):
            return False
        if self._random_count % self._light_random_cycle == 0:
            return True
        return False

    def spawn_lights(self):
        light_config = self._config.get("light", {})
        light_names = light_config.get("names", [])
        light_spawnable = light_config.get("spawnable", [])

        light_random = light_config.get("random", {})
        is_random_position = light_random.get("position", False)
        is_random_rotation = light_random.get("rotation", False)

        if not (is_random_position or is_random_rotation):
            return

        light_random_nums = light_random.get("nums")
        light_random_center = light_random.get("center")
        light_random_bound_position = light_random.get("bound_position", [[0, 0], [0, 0], [0, 0]])
        bound_rotation = light_random.get("bound_rotation", [[0, 0], [0, 0], [0, 0]])
        bound_position = [[light_random_center[0] + light_random_bound_position[0][0], light_random_center[0] + light_random_bound_position[0][1]], 
                          [light_random_center[1] + light_random_bound_position[1][0], light_random_center[1] + light_random_bound_position[1][1]], 
                          [light_random_center[2] + light_random_bound_position[2][0], light_random_center[2] + light_random_bound_position[2][1]]]
       
        light_index = choose_random_indices(len(light_names), light_random_nums)
        for i in light_index:
            light_name = light_names[i]
            light_spawnable = light_spawnable[i]
            transform = get_random_transform(bound_position, bound_rotation)
            self.add_light(light_name, light_spawnable, transform[:3], transform[3:])

    def publish_scene_without_init_env(self):
        self._scene.publish_scene()    

    def publish_scene(self):
        """
        Publish the scene to the ORCA Gym environment.
        """
        self._scene.publish_scene()
        time.sleep(3)
        if self._init_env_callback is not None:
            self._init_env_callback()
        else:
            orca_log.warning("init_env_callback is not set")

    def add_actor(self, actor_name: str, asset_path: str, position: np.ndarray, rotation: np.ndarray, scale: float = 1.0):
        actor = Actor(actor_name, asset_path, position, rotation, scale)
        self._scene.add_actor(actor)

    def add_light(self, light_name: str, asset_path: str, position: np.ndarray, rotation: np.ndarray, scale: float = 1.0):
        actor = Actor(light_name, asset_path, position, rotation, scale)
        self._scene.add_actor(actor)

    
    def _config_check(self):
        '''
        check the config is valid
        '''
        actor_config = self._config.get("actor", {})
        actor_names = actor_config.get("names", [])
        actor_spawnable = actor_config.get("spawnable", [])
        actor_joints_dof = actor_config.get("joints_dof", [])
        actor_joints = actor_config.get("joints", [])

        if len(actor_names) != len(actor_spawnable):
            orca_log.error(f'''Has {len(actor_names)} actors and {len(actor_spawnable)} spawnables, 
            The number of actor names and spawnable must be the same.''')
            raise ValueError("The number of actor names and spawnable must be the same.")
        

        if len(actor_names) != len(actor_joints):
            orca_log.error("The number of actor names and joints must be the same.")
            raise ValueError("The number of actor names and joints must be the same.")

        actor_joints_dof = actor_config.get("joints_dof", [])
        if len(actor_names) != len(actor_joints_dof):
            orca_log.error("The number of actor names and joints_dof must be the same.")
            raise ValueError("The number of actor names and joints_dof must be the same.")
        
        actor_random = actor_config.get("random", {})
        actor_random_qpos = actor_random.get("qpos", False)
        actor_random_nums = actor_random.get("nums", [0, 0])
        actor_random_six_dof = actor_random.get("six_dof", None)
        actor_random_three_dof = actor_random.get("three_dof", None)
        actor_random_one_dof = actor_random.get("one_dof", None)

        if actor_random_qpos:
            if not (actor_random_nums[0] > 0 
                and actor_random_nums[1] > actor_random_nums[0]
                and actor_random_nums[1] <= len(actor_names)):
                orca_log.error("The actor.random.nums is invalid, the first number must be greater than 0, the second number must be greater than the first number, and the second number must be less than the number of actors.")
                raise ValueError("The actor.random.nums is invalid, the first number must be greater than 0, the second number must be greater than the first number, and the second number must be less than the number of actors.")
            
        if 6 in actor_joints_dof:
            if actor_random_six_dof is None:
                orca_log.error("actor.random.six_dof is not set")
                orca_log.error('''example:
                                actor:
                                  random:
                                    six_dof:
                                      center: [0, 0, 0]
                                      bound_position: [[-1, 1], [-1, 1], [0, 2]]
                                      bound_rotation: [[0, 3.14159], [0, 3.14159], [0, 3.14159]]
                                ''')
                raise ValueError("actor.random.six_dof is not set")
            else:
                if actor_random_six_dof.get("bound_position", None) is None:
                    orca_log.error("actor.random.six_dof.bound_position is not set")
                    orca_log.error('''example:
                                    actor:
                                      random:
                                        six_dof:
                                          bound_position: [[-1, 1], [-1, 1], [0, 2]]
                                          bound_rotation: [[0, 3.14159], [0, 3.14159], [0, 3.14159]]
                                    ''')
                    raise ValueError("actor.random.six_dof.bound_position is not set")
                if actor_random_six_dof.get("center", None) is None:
                    orca_log.error("actor.random.six_dof.center is not set")
                    orca_log.error('''example:
                                    actor:
                                      random:
                                        six_dof:
                                          center: [0, 0, 0]
                                    ''')
                    raise ValueError("actor.random.six_dof.center is not set")
        if 3 in actor_joints_dof:
            if actor_random_three_dof is None or actor_random_three_dof.get("center", None) is None:
                orca_log.error("actor.random.three_dof is not set or actor.random.three_dof.center is not set")
                orca_log.error('''example:
                                    actor:
                                      random:
                                        three_dof:
                                          center: [0, 0, 0]
                                          bound: [0, 1]
                                    ''')
                raise ValueError("actor.random.three_dof is not set or actor.random.three_dof.center is not set")
            else:
                if actor_random_three_dof.get("bound", None) is None:
                    orca_log.error("actor.random.three_dof.bound is not set")
                    orca_log.error('''example:
                                    actor:
                                      random:
                                        three_dof:
                                          bound: [0, 1]
                                    ''')
                    raise ValueError("actor.random.three_dof.bound is not set")

        if 1 in actor_joints_dof:
            if actor_random_one_dof is None or actor_random_one_dof.get("center", None) is None:
                orca_log.error("actor.random.one_dof is not set")
                orca_log.error('''example:
                                actor:
                                  random:
                                    one_dof:
                                      bound: [-1, 1]
                                    ''')
                raise ValueError("actor.random.one_dof is not set or actor.random.one_dof.center is not set")
            else:
                if actor_random_one_dof.get("bound", None) is None:
                    orca_log.error("actor.random.one_dof.bound is not set")
                    orca_log.error('''example:
                                    actor:
                                      random:
                                        one_dof:
                                          bound: [-1, 1]
                                    ''')
                    raise ValueError("actor.random.one_dof.bound is not set")


        light_config = self._config.get("light", {})
        light_names = light_config.get("names", [])
        light_spawnable = light_config.get("spawnable", [])
        light_random = light_config.get("random", None)
        if len(light_names) != len(light_spawnable):
            orca_log.error(f'''Has {len(light_names)} lights and {len(light_spawnable)} spawnables, 
            The number of light names and spawnable must be the same.''')
            raise ValueError("The number of light names and spawnable must be the same.")
        
        if light_random is not None:
            is_random_position = light_random.get("position", False)
            is_random_rotation = light_random.get("rotation", False)
            light_random_center = light_random.get("center", None)            
            light_random_bound_position = light_random.get("bound_position", None)
            light_random_bound_rotation = light_random.get("bound_rotation", None)
            light_random_nums = light_random.get("nums", [0, 0])   
            light_random_cycle = light_random.get("cycle", 20)

            if is_random_position and light_random_bound_position is None:
                orca_log.error("light.random.bound_position is not set")
                orca_log.error('''example:
                                light:
                                  random:
                                    center: [0, 0, 0]
                                    position: true
                                    bound_position: [[-1, 1], [-1, 1], [0, 2]]
                                ''')
                raise ValueError("light.random.bound_position is not set")

            if is_random_rotation and light_random_bound_rotation is None:
                orca_log.error("light.random.bound_rotation is not set")
                orca_log.error('''example:
                                light:
                                  random:
                                    center: [0, 0, 0]
                                    rotation: true
                                    bound_rotation: [[0, 3.14159], [0, 3.14159], [0, 3.14159]]
                                ''')
                raise ValueError("light.random.bound_rotation is not set")

            if light_random_center is None:
                orca_log.error("light.random.center is not set")
                orca_log.error('''example:
                                light:
                                  random:
                                    position: true
                                    center: [0, 0, 0]
                                    bound_position: [[-1, 1], [-1, 1], [0, 2]]
                                ''')
                raise ValueError("light.random.center is not set")

            if not (light_random_nums[0] > 0 
                and light_random_nums[1] >= light_random_nums[0]
                and light_random_nums[1] <= len(light_names)):
                orca_log.error("The light.random.nums is invalid, the first number must be greater than 0, the second number must be greater than the first number, and the second number must be less than the number of lights.")
                raise ValueError("The light.random.nums is invalid, the first number must be greater than 0, the second number must be greater than the first number, and the second number must be less than the number of lights.")
        
            if light_random_cycle <= 0:
                orca_log.error("light.random.cycle is invalid, the cycle must be greater than 0")
                orca_log.error('''example:
                                light:
                                  random:
                                    cycle: 20
                                ''')
                raise ValueError("light.random.cycle is invalid, the cycle must be greater than 0")
        else:
            orca_log.error("light.random is not set")
            orca_log.error('''example:
                            light:
                              random:
                                position: true
                                rotation: true
                                center: [0, 0, 0]
                                bound_position: [[-1, 1], [-1, 1], [0, 2]]
                                bound_rotation: [[0, 3.14159], [0, 3.14159], [0, 3.14159]]
                                nums: [3, 5]
                            ''')
            raise ValueError("light.random is not set")