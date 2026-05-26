"""
MuJoCo to SPH Scene Generator

该模块提供从 MuJoCo 模型自动生成 SPH scene.json (RigidBodies 部分) 的功能。
支持坐标系转换、几何映射推断和 SPH 标记识别，确保两个仿真系统初始化的一致性。
"""

import json
import os
import numpy as np
import logging
from pathlib import Path

from ..paths import FLUID_PACKAGE_DIR
from ..launch.sph_config import _deep_merge
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys
import mujoco

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[SceneGenerator] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


@dataclass
class GeomInfo:
    """几何信息数据类"""
    geom_type: str
    mesh_name: str = ""
    size: List[float] = None
    scale: List[float] = None
    is_static: bool = False  # 标记是否为静态刚体
    
    def __post_init__(self):
        if self.size is None:
            self.size = []
        if self.scale is None:
            self.scale = [1, 1, 1]


class SceneGenerator:
    """MuJoCo to SPH 场景生成器"""
    
    def __init__(self, env, config: Dict = None, config_path: str = None, runtime_config: Dict = None):
        """
        初始化 SceneGenerator
        
        Args:
            env: OrcaGymLocalEnv 实例
            config: 配置字典（场景模板配置，直接传入）
            config_path: 场景模板配置文件路径（优先从路径加载）
            runtime_config: 运行时配置（必须包含 orcalink_bridge.shared_modules.spring_force）
        """
        self.env = env
        self.config = self._load_config(config_path) if config_path else config or {}
        self.runtime_config = runtime_config or {}

        # particleRadius — authoritative value used throughout scene generation.
        # Initialized from scene_config.json as a safe default; overridden early in
        # generate_complete_scene() by reading the MJCF bound site User[0]/User[1]
        # (Orca: radius + PRBTransmissionMode tag for particle_render.grpc merge).
        self.particle_radius: float = float(
            self.config.get("scene_template", {})
                        .get("Configuration", {})
                        .get("particleRadius", 0.025)
        )
        # MJCF *_SPH_PARTICLE_RENDER_BOUNDS site User[1] (PRBTransmissionMode), if present
        self.particle_render_transmission_tag: Optional[int] = None

        # 保存 scene 文件所在的目录（用于相对路径转换）
        self.scene_dir = None  # 在 generate_complete_scene 中设置
        
        # 缓存 SPH body 的 geom 信息，避免重复查找
        # 格式: {body_name: GeomInfo}
        self._sph_geom_cache = {}
        
        # 基准目录：envs.fluid 包根（与原先 scene_generator 位于 fluid/ 下时一致）
        self._base_dir = str(FLUID_PACKAGE_DIR)
        
        logger.info(f"SceneGenerator initialized")
    
    def _resolve_geometry_path(self, geometry_file: str) -> str:
        """
        将相对路径转换为绝对路径，支持包资源加载
        
        支持三种路径格式：
        1. 绝对路径：直接返回
        2. package://orcasph/data/models/UnitBox.obj：从 orcasph_client 包加载
        3. 相对路径：相对于 envs.fluid 包根目录，如果不存在则尝试从包中加载
        
        Args:
            geometry_file: 几何文件路径（可能是相对路径或绝对路径）
            
        Returns:
            绝对路径
        """
        if not geometry_file:
            return geometry_file
            
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(geometry_file):
            return geometry_file
        
        # 支持包资源路径：package://orcasph/data/models/UnitBox.obj
        if geometry_file.startswith('package://orcasph/'):
            resource_path = geometry_file[len('package://orcasph/'):]
            abs_path = self._load_from_orcasph_package(resource_path)
            if abs_path:
                logger.info(f"Loaded from orcasph package: '{geometry_file}' -> '{abs_path}'")
                return abs_path
            else:
                logger.warning(f"Failed to load package resource: '{geometry_file}'")
        
        # 相对路径：从 envs.fluid 包根目录解析
        abs_path = os.path.abspath(os.path.join(self._base_dir, geometry_file))
        
        # 如果文件不存在，尝试从 orcasph_client 包中加载（fallback）
        if not os.path.exists(abs_path):
            logger.warning(f"File not found at: {abs_path}")
            
            # 尝试从包中加载
            # 例如：../../../data/models/UnitBox.obj -> data/models/UnitBox.obj
            filename = os.path.basename(geometry_file)
            dirname = os.path.dirname(geometry_file)
            
            # 尝试常见的资源路径
            possible_paths = [
                f"data/models/{filename}",
                f"data/{filename}",
                filename,
            ]
            
            for resource_path in possible_paths:
                package_path = self._load_from_orcasph_package(resource_path)
                if package_path and os.path.exists(package_path):
                    logger.info(f"Found in orcasph package: '{geometry_file}' -> '{package_path}'")
                    return package_path
            
            logger.warning(f"Could not find file in package, using original path: {abs_path}")
        
        logger.info(f"Resolved geometry path: '{geometry_file}' -> '{abs_path}'")
        return abs_path
    
    def _load_from_orcasph_package(self, resource_path: str) -> str:
        """
        从 orcasph_client 包中加载资源文件
        
        兼容：
        - pip install orca-sph（普通安装）
        - pip install -e .（可编辑安装）
        
        Args:
            resource_path: 包内资源路径，如 'data/models/UnitBox.obj'
            
        Returns:
            资源文件的绝对路径，如果找不到返回 None
        """
        try:
            # 方法1：使用 __file__ 定位（兼容 editable install）
            try:
                import orcasph_client
                package_dir = Path(orcasph_client.__file__).parent
                resource_file = package_dir / resource_path
                
                if resource_file.exists():
                    return str(resource_file.resolve())
            except ImportError:
                pass
            
            # 方法2：使用 importlib.resources（Python 3.9+）
            try:
                from importlib import resources
                if hasattr(resources, 'files'):
                    resource = resources.files('orcasph_client').joinpath(resource_path)
                    if resource.is_file():
                        return str(resource)
            except Exception:
                pass
            
            # 方法3：使用 pkg_resources（fallback，兼容旧版本）
            try:
                import pkg_resources
                resource_file = pkg_resources.resource_filename('orcasph_client', resource_path)
                if os.path.exists(resource_file):
                    return resource_file
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error loading package resource '{resource_path}': {e}")
            return None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载 JSON 配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file: {e}, using defaults")
            return {}
    
    def identify_sph_bodies(self) -> List[str]:
        """
        识别需要导出的刚体（有 SPH_MESH_GEOM 或 SPH_STATIC_MESH_GEOM 的 body）并缓存 geom 信息
        
        识别策略：
        1. 遍历所有 geom，根据名称过滤出 SPH_MESH_GEOM 和 SPH_STATIC_MESH_GEOM
        2. 从 geom 的 BodyName 字段直接获取 body 名称
        3. 同时获取 mesh 文件路径和 scale 信息并缓存
        4. 去重并返回
        
        命名规则：
        - 动态刚体：${bodyName}_SPH_MESH_GEOM，对应 mesh 为 ${bodyName}_SPH_MESH
        - 静态刚体：${bodyName}_SPH_STATIC_MESH_GEOM，对应 mesh 为 ${bodyName}_SPH_STATIC_MESH
        
        Returns:
            List[str]: body 名称列表
        """
        sph_bodies = set()
        self._sph_geom_cache.clear()  # 清空缓存
        
        try:
            model = self.env.model
            
            # 获取 geom 字典和 mesh 字典
            geom_dict = model.get_geom_dict()
            mesh_dict = model.get_mesh_dict()
            
            if not geom_dict:
                logger.warning("geom_dict not available")
                return []
            
            logger.info(f"Scanning {len(geom_dict)} geoms for SPH geoms (dynamic and static)...")
            
            # 过滤出 SPH_MESH_GEOM 和 SPH_STATIC_MESH_GEOM 并提取信息
            sph_geoms = []
            for geom_name, geom_info in geom_dict.items():
                # 检查是否为静态刚体或动态刚体
                is_static = '_SPH_STATIC_MESH_GEOM' in geom_name
                is_dynamic = '_SPH_MESH_GEOM' in geom_name and not is_static
                
                if is_static or is_dynamic:
                    body_name = geom_info.get('BodyName', '')
                    if not body_name:
                        logger.warning(f"SPH geom '{geom_name}' has no BodyName!")
                        continue
                    
                    sph_bodies.add(body_name)
                    sph_geoms.append({
                        'geom': geom_name, 
                        'body': body_name, 
                        'is_static': is_static
                    })
                    
                    # 获取并缓存 geom 信息（mesh 文件路径和 scale）
                    # 注意：
                    # 1. 不能通过 geom 的 DataID 获取，因为 DataID 可能指向其他用途的 mesh
                    # 2. 应该直接通过命名规则获取 SPH_MESH 或 SPH_STATIC_MESH
                    # 3. scale 从 mesh 的 Scale 属性获取（从 XML 解析）
                    try:
                        if is_static:
                            sph_mesh_name = f"{body_name}_SPH_STATIC_MESH"
                        else:
                            sph_mesh_name = f"{body_name}_SPH_MESH"
                        
                        mesh_info = mesh_dict.get(sph_mesh_name) if mesh_dict else None

                        if mesh_info is None and mesh_dict:
                            dataid = int(geom_info.get('DataID', -1))
                            if dataid >= 0:
                                for mname, minfo in mesh_dict.items():
                                    if int(minfo.get('ID', -1)) == dataid:
                                        mesh_info = minfo
                                        sph_mesh_name = mname
                                        logger.info(f"Resolved mesh for '{body_name}' via DataID={dataid}: '{mname}'")
                                        break
                        
                        if mesh_info:
                            mesh_file = mesh_info.get('File', '')
                            mesh_scale = mesh_info.get('Scale', [1.0, 1.0, 1.0])
                            
                            self._sph_geom_cache[body_name] = GeomInfo(
                                geom_type='mesh',
                                mesh_name=mesh_file,
                                size=[],
                                scale=[float(x) for x in mesh_scale] if mesh_scale else [1.0, 1.0, 1.0],
                                is_static=is_static
                            )
                            
                            body_type = "static" if is_static else "dynamic"
                            logger.info(f"Cached {body_type} geom info for '{body_name}': mesh='{sph_mesh_name}', file='{mesh_file}', scale={mesh_scale}")
                        else:
                            logger.warning(f"❌ Cannot resolve mesh for body '{body_name}', removing from SPH bodies")
                            sph_bodies.discard(body_name)
                            sph_geoms = [g for g in sph_geoms if g['body'] != body_name]
                    except Exception as e:
                        logger.warning(f"❌ Failed to cache geom info for '{body_name}': {e}", exc_info=True)
                        sph_bodies.discard(body_name)
                        sph_geoms = [g for g in sph_geoms if g['body'] != body_name]
            
            # 日志输出识别结果
            if sph_geoms:
                dynamic_count = sum(1 for g in sph_geoms if not g['is_static'])
                static_count = sum(1 for g in sph_geoms if g['is_static'])
                logger.info(f"Found {len(sph_geoms)} SPH geom(s): {dynamic_count} dynamic, {static_count} static")
                for item in sph_geoms:
                    body_type = "static" if item['is_static'] else "dynamic"
                    logger.info(f"  - [{body_type}] geom: '{item['geom']}' -> body: '{item['body']}'")
            
            result = sorted(list(sph_bodies))
            logger.info(f"Identified {len(result)} SPH body(bodies): {result}")
            logger.info(f"Cached geom info for {len(self._sph_geom_cache)} bodies")
            return result
            
        except Exception as e:
            logger.error(f"Error identifying SPH bodies: {e}", exc_info=True)
            return []

    def identify_fluid_block_geoms(self) -> List[Dict]:
        """
        识别流体块几何体（命名包含 _SPH_FLUID_BLOCK_GEOM 的 geom）。

        当 geom 类型为 "box" 时（无 wall mesh asset）：
          从 geom 的 Pos（中心）、Size（半轴）重建 start/end。

        当 geom 类型为 "mesh" 时（有 wall mesh asset，MjSphFluidBlockComponent 写入）：
          - start/end 从 _SPH_FLUID_BLOCK_BOUNDS site 中读取（C++ 同步写入）
          - wall_mesh_file 从 mesh_dict 中查找 ${body_name}_SPH_FLUID_BLOCK_WALL_MESH 获取

        全部为 MuJoCo Z-up 原始数据，不做坐标系转换；转换在 generate_complete_scene 写 scene.json 时统一完成。

        Returns:
            List[Dict]: 每个元素包含 geom_name, body_name, start, end, position（均为 MuJoCo Z-up）
                        以及可选字段 wall_mesh_file（str，当有 wall mesh asset 时填入绝对路径）
        """
        fluid_blocks = []
        try:
            model = self.env.model
            geom_dict = model.get_geom_dict()
            if not geom_dict:
                return fluid_blocks
            mesh_dict = model.get_mesh_dict() if hasattr(model, 'get_mesh_dict') else {}
            site_dict = model.get_site_dict() if hasattr(model, 'get_site_dict') else {}
            for geom_name, geom_info in geom_dict.items():
                if '_SPH_FLUID_BLOCK_GEOM' not in geom_name:
                    continue
                body_name = geom_info.get('BodyName', '')
                if not body_name:
                    logger.warning(f"Fluid block geom '{geom_name}' has no BodyName")
                    continue

                geom_type = geom_info.get('Type', mujoco.mjtGeom.mjGEOM_BOX)
                wall_mesh_file = None

                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    logger.info(f"Fluid block geom '{geom_name}' is type=mesh")
                    # Wall mesh asset was set in MjSphFluidBlockComponent.
                    # start/end come from the companion _SPH_FLUID_BLOCK_BOUNDS site.
                    bounds_site_name = body_name + '_SPH_FLUID_BLOCK_BOUNDS'
                    bounds_info = site_dict.get(bounds_site_name) if site_dict else None
                    if bounds_info is None:
                        logger.warning(
                            f"Fluid block geom '{geom_name}' is type=mesh but bounds site "
                            f"'{bounds_site_name}' not found; skipping")
                        continue
                    size_arr = bounds_info.get('Size', None)
                    pos_arr = bounds_info.get('Pos', None)
                    if size_arr is None or pos_arr is None:
                        logger.warning(
                            f"Bounds site '{bounds_site_name}' missing Size or Pos; skipping")
                        continue

                    # Resolve wall mesh file path from mesh_dict
                    wall_mesh_asset_name = body_name + '_SPH_FLUID_BLOCK_WALL_MESH'
                    mesh_info = mesh_dict.get(wall_mesh_asset_name) if mesh_dict else None
                    if mesh_info:
                        wall_mesh_file = mesh_info.get('File', None)
                        logger.info(
                            f"Wall mesh asset for '{body_name}': "
                            f"name='{wall_mesh_asset_name}', file='{wall_mesh_file}'")
                    else:
                        logger.warning(
                            f"Wall mesh '{wall_mesh_asset_name}' not found in mesh_dict for "
                            f"fluid block body '{body_name}'")
                else:
                    logger.info(f"Fluid block geom '{geom_name}' is type=box")
                    # type="box" (no wall mesh asset): read size/pos directly from geom
                    size_arr = geom_info.get('Size', None)
                    pos_arr = geom_info.get('Pos', None)
                    if size_arr is None or pos_arr is None:
                        logger.warning(f"Fluid block geom '{geom_name}' missing Size or Pos")
                        continue

                half = np.array(size_arr[:3], dtype=float)
                center = np.array(pos_arr[:3], dtype=float)
                start = (center - half).tolist()
                end = (center + half).tolist()
                # 直接查询 body 世界坐标，不依赖 _sph_geom_cache（FluidBlock body 无 SPH connector geom）
                xpos_flat, _, _ = self.env.get_body_xpos_xmat_xquat([body_name])
                if xpos_flat is not None and len(xpos_flat) >= 3:
                    position = np.array(xpos_flat[:3])
                else:
                    logger.warning(f"Could not get position for body '{body_name}', using zero")
                    position = np.zeros(3)
                entry = {
                    'geom_name': geom_name,
                    'body_name': body_name,
                    'start': start,
                    'end': end,
                    'position': position,
                }
                if wall_mesh_file:
                    entry['wall_mesh_file'] = wall_mesh_file
                fluid_blocks.append(entry)
                logger.info(
                    f"Fluid block (Z-up): body={body_name}, start={start}, end={end}, "
                    f"position={position}, wall_mesh_file={wall_mesh_file}")
        except Exception as e:
            logger.error(f"Error identifying fluid block geoms: {e}", exc_info=True)
        return fluid_blocks

    def extract_mocap_bodies_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体对应的所有 Mocap site 世界坐标位置
        
        识别策略：
        1. 遍历所有 site
        2. 查找包含 "SPH_MOCAP_SITE" 且属于该 body 的 site
        3. 使用 query_site_pos_and_mat 读取世界坐标位置
        
        Args:
            body_name: 主刚体名称，例如 "toys_usda_box_body"
        
        Returns:
            List of dicts with keys:
                - 'mocap_body_name': 完整的 mocap body 名称
                - 'index': mocap 索引 (0-3)
                - 'world_position': 世界坐标位置 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        mocap_info_list = []
        
        try:
            model = self.env.model
            
            # 1. 获取所有 site 的字典
            site_dict = model.get_site_dict()
            if not site_dict:
                logger.warning(f"site_dict not available for body '{body_name}'")
                return []
            
            # 2. 筛选出属于该主刚体的 mocap site
            # 命名模式: "{body_name}_SPH_MOCAP_SITE_{index:03d}"
            prefix = f"{body_name}_SPH_MOCAP_SITE_"
            
            # 收集符合条件的 site 名称
            site_names = []
            for site_name in site_dict.keys():
                if site_name.startswith(prefix):
                    site_names.append(site_name)
            
            if not site_names:
                logger.debug(f"No SPH_MOCAP_SITE found for body '{body_name}'")
                return []
            
            # 3. 使用 query_site_pos_and_mat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            site_pos_data = self.env.query_site_pos_and_mat(site_names)
            
            # 4. 构造返回列表，包含 world_position
            for site_name in site_names:
                # 提取索引
                try:
                    index_str = site_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置
                if site_name in site_pos_data:
                    xpos = site_pos_data[site_name]['xpos']
                    # 转换为列表格式
                    if isinstance(xpos, np.ndarray):
                        world_pos = xpos[:3].tolist()
                    elif isinstance(xpos, (list, tuple)) and len(xpos) >= 3:
                        world_pos = list(xpos[:3])
                    else:
                        logger.warning(f"Invalid xpos format for site '{site_name}': {xpos}")
                        world_pos = [0.0, 0.0, 0.0]
                else:
                    logger.warning(f"Site '{site_name}' not found in query_site_pos_and_mat result")
                    world_pos = [0.0, 0.0, 0.0]
                
                # 构造对应的 mocap body 名称
                mocap_body_name = f"{body_name}_SPH_MOCAP_{index:03d}"
                
                mocap_info_list.append({
                    'mocap_body_name': mocap_body_name,
                    'index': index,
                    'world_position': world_pos  # MuJoCo 世界坐标 (Z-up)
                })
            
            # 5. 按索引排序
            mocap_info_list.sort(key=lambda x: x['index'])
            
            logger.debug(f"Extracted {len(mocap_info_list)} mocap bodies for '{body_name}'")
            return mocap_info_list
            
        except Exception as e:
            logger.error(f"Error extracting mocap bodies for '{body_name}': {e}", exc_info=True)
            return []
    
    def convert_local_coord_z_to_y(self, local_pos_mj: np.ndarray) -> List[float]:
        """
        转换本地坐标从 MuJoCo Z-up 到 SPH Y-up
        
        坐标系转换规则：
            MuJoCo Z-up: X right, Y forward, Z up
            SPH Y-up:    X right, Z forward, Y up
        
        转换公式：[x, y, z] → [x, z, -y]
        
        Args:
            local_pos_mj: MuJoCo 本地坐标 [x, y, z]
        
        Returns:
            SPH 本地坐标 [x, z, -y]
        """
        return [float(local_pos_mj[0]), float(local_pos_mj[2]), -float(local_pos_mj[1])]
    
    def extract_site_local_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_SITE 本地坐标位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts:
                - 'site_name': 完整的 site 名称
                - 'index': site 索引
                - 'local_position': 本地坐标 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        site_info_list = []
        
        try:
            # 从 model 获取所有 site 的字典（包含 LocalPos）
            site_dict = self.env.model.get_site_dict()
            prefix = f"{body_name}_SPH_SITE_"
            
            for site_name, site_data in site_dict.items():
                if site_name.startswith(prefix):
                    # 提取索引
                    try:
                        index_str = site_name.replace(prefix, "")
                        index = int(index_str)
                    except ValueError:
                        continue
                    
                    # 直接读取本地坐标（已经是相对于 body 的坐标）
                    local_pos = site_data.get('LocalPos', np.array([0.0, 0.0, 0.0]))
                    if isinstance(local_pos, np.ndarray):
                        local_pos = local_pos.tolist()
                    elif isinstance(local_pos, (list, tuple)):
                        local_pos = list(local_pos[:3])
                    else:
                        logger.warning(f"Invalid LocalPos format for site '{site_name}': {local_pos}")
                        local_pos = [0.0, 0.0, 0.0]
                    
                    site_info_list.append({
                        'site_name': site_name,
                        'index': index,
                        'local_position': local_pos  # MuJoCo 本地坐标 (Z-up)
                    })
            
            site_info_list.sort(key=lambda x: x['index'])
            logger.debug(f"Extracted {len(site_info_list)} site local positions for '{body_name}'")
            return site_info_list
            
        except Exception as e:
            logger.error(f"Error extracting site local positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_mocap_world_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_MOCAP_SITE 全局坐标位置
        
        注意：MOCAP body 是独立的 mocap body，不隶属于主刚体
        需要从 body_xpos 读取全局位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts:
                - 'mocap_name': 完整的 mocap body 名称
                - 'index': mocap body 索引
                - 'world_position': 全局坐标 [x, y, z] (SPH Y-up 坐标系)
        """
        mocap_info_list = []
        
        try:
            # 1. 从 model 获取所有 body 的字典，筛选出属于该主刚体的 SPH_MOCAP body
            body_dict = self.env.model.get_body_dict()
            prefix = f"{body_name}_SPH_MOCAP_"
            
            # 收集符合条件的 mocap body 名称
            mocap_body_names = []
            for body_name_mj in body_dict.keys():
                if body_name_mj.startswith(prefix):
                    mocap_body_names.append(body_name_mj)
            
            if not mocap_body_names:
                logger.debug(f"No SPH_MOCAP body found for '{body_name}'")
                return []
            
            # 2. 使用 get_body_xpos_xmat_xquat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            xpos_flat, _, _ = self.env.get_body_xpos_xmat_xquat(mocap_body_names)
            
            # 3. 构造返回列表，包含 world_position (MuJoCo 坐标系，未转换)
            for i, mocap_body_name in enumerate(mocap_body_names):
                # 提取索引
                try:
                    index_str = mocap_body_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置（xpos_flat 是扁平化的数组，每个 body 3个元素）
                world_pos_mj = xpos_flat[i*3:(i+1)*3]
                
                mocap_info_list.append({
                    'mocap_name': mocap_body_name,
                    'index': index,
                    'world_position': world_pos_mj.tolist() if isinstance(world_pos_mj, np.ndarray) else list(world_pos_mj)
                })
            
            mocap_info_list.sort(key=lambda x: x['index'])
            logger.debug(f"Extracted {len(mocap_info_list)} mocap world positions for '{body_name}'")
            return mocap_info_list
            
        except Exception as e:
            logger.error(f"Error extracting mocap world positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_site_positions_for_body(self, body_name: str) -> List[Dict]:
        """
        提取指定主刚体的所有 SPH_SITE 世界坐标位置
        
        Args:
            body_name: 主刚体名称
        
        Returns:
            List of dicts with keys:
                - 'site_name': 完整的 site 名称
                - 'index': site 索引 (0-3)
                - 'world_position': 世界坐标位置 [x, y, z] (MuJoCo Z-up 坐标系)
        """
        site_info_list = []
        
        try:
            # 1. 从 model 获取所有 site 的字典，筛选出属于该主刚体的 SPH_SITE
            site_dict = self.env.model.get_site_dict()
            prefix = f"{body_name}_SPH_SITE_"
            
            # 收集符合条件的 site 名称
            site_names = []
            for site_name in site_dict.keys():
                if site_name.startswith(prefix):
                    site_names.append(site_name)
            
            if not site_names:
                logger.debug(f"No SPH_SITE found for body '{body_name}'")
                return []
            
            # 2. 使用 query_site_pos_and_mat 获取世界坐标（需要先调用 mj_forward 更新数据）
            self.env.mj_forward()
            site_pos_data = self.env.query_site_pos_and_mat(site_names)
            
            # 3. 构造返回列表，包含 world_position
            for site_name in site_names:
                # 提取索引
                try:
                    index_str = site_name.replace(prefix, "")
                    index = int(index_str)
                except ValueError:
                    continue
                
                # 获取世界坐标位置
                if site_name in site_pos_data:
                    xpos = site_pos_data[site_name]['xpos']
                    # 转换为列表格式
                    if isinstance(xpos, np.ndarray):
                        world_pos = xpos[:3].tolist()
                    elif isinstance(xpos, (list, tuple)) and len(xpos) >= 3:
                        world_pos = list(xpos[:3])
                    else:
                        logger.warning(f"Invalid xpos format for site '{site_name}': {xpos}")
                        world_pos = [0.0, 0.0, 0.0]
                else:
                    logger.warning(f"Site '{site_name}' not found in query_site_pos_and_mat result")
                    world_pos = [0.0, 0.0, 0.0]
                
                site_info_list.append({
                    'site_name': site_name,
                    'index': index,
                    'world_position': world_pos  # MuJoCo 世界坐标 (Z-up)
                })
            
            # 4. 按索引排序
            site_info_list.sort(key=lambda x: x['index'])
            
            logger.debug(f"Extracted {len(site_info_list)} site positions for '{body_name}'")
            return site_info_list
            
        except Exception as e:
            logger.error(f"Error extracting site positions for '{body_name}': {e}", exc_info=True)
            return []
    
    def extract_body_info(self, body_name: str) -> Dict:
        """
        从 OrcaGym model 提取 body 完整信息
        
        Args:
            body_name: MuJoCo body 名称
            
        Returns:
            Dict: body 信息字典
        """
        try:
            model = self.env.model
            
            # 获取 body ID
            body_id = model.body_name2id(body_name)
            
            # 获取位置和旋转 - 返回扁平数组，需要提取单个 body 数据
            xpos_flat, _, xquat_flat = self.env.get_body_xpos_xmat_xquat([body_name])
            
            # 提取单个 body 的数据
            if xpos_flat is not None and len(xpos_flat) >= 3:
                xpos = np.array(xpos_flat[:3])
            else:
                xpos = np.array([0.0, 0.0, 0.0])
            
            if xquat_flat is not None and len(xquat_flat) >= 4:
                xquat = np.array(xquat_flat[:4])
            else:
                xquat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # 获取质量
            body_dict = model.get_body_dict()
            body_data = body_dict.get(body_name, {})
            mass = body_data.get('Mass', 1.0)
            subtree_mass = body_data.get('SubtreeMass', 1.0)
            inertia = body_data.get('Inertia', [0, 0, 0])
            
            # 调试：打印质量相关信息
            logger.info(f"[DEBUG] Body '{body_name}' mass info:")
            logger.info(f"  Mass: {mass}")
            logger.info(f"  SubtreeMass: {subtree_mass}")
            logger.info(f"  Inertia: {inertia}")
            
            # 从缓存获取 geom 信息（在 identify_sph_bodies 中已经获取并缓存）
            geom_info = self._sph_geom_cache.get(body_name)
            if geom_info is None:
                error_msg = (
                    f"Geom info not found in cache for body '{body_name}'. "
                    f"Please ensure identify_sph_bodies() is called before extract_body_info()."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Extracted body info for '{body_name}': pos={xpos}, quat={xquat}, mass={mass}")
            
            return {
                'body_id': body_id,
                'body_name': body_name,
                'position': xpos,
                'quaternion': xquat,
                'mass': mass,
                'geom_info': geom_info
            }
            
        except Exception as e:
            error_msg = f"Error extracting body info for '{body_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def convert_coordinate_z_to_y(self, pos: np.ndarray, quat: np.ndarray) -> Tuple[List, List, float]:
        """
        坐标系转换：MuJoCo Z-up → SPH Y-up
        
        坐标系定义：
            MuJoCo Z-up: X right, Y forward, Z up
            SPH Y-up:    X right, Z forward, Y up
            
        转换公式：
            位置: [x, y, z] → [x, z, -y]
            旋转: 绕 X 轴旋转 -90 度
        
        Args:
            pos: 位置 [x, y, z]
            quat: 四元数 [w, x, y, z]
            
        Returns:
            Tuple: (translation, rotationAxis, rotationAngle)
        """
        try:
            from scipy.spatial.transform import Rotation as R
            
            # 位置转换：MuJoCo Z-up [x, y, z] → SPH Y-up [x, z, -y]
            translation = [float(pos[0]), float(pos[2]), -float(pos[1])]
            
            # 四元数转换（MuJoCo 格式 [w, x, y, z] → SciPy 格式 [x, y, z, w]）
            rot = R.from_quat([float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
            
            # 应用坐标系变换矩阵（绕X轴旋转-90度）
            # 从 Z-up 到 Y-up：Y → -Z, Z → Y，相当于绕 X 轴旋转 -90 度
            transform = R.from_euler('x', -90, degrees=True)
            rot_transformed = transform * rot
            
            # 转换为轴角表示
            rotvec = rot_transformed.as_rotvec()
            angle = float(np.linalg.norm(rotvec))
            
            if angle > 1e-6:
                axis = (rotvec / angle).tolist()
            else:
                axis = [1.0, 0.0, 0.0]
            
            logger.debug(f"Coordinate converted: pos={pos} → {translation}, quat={quat} → axis={axis}, angle={angle}")
            
            return translation, axis, angle
            
        except Exception as e:
            logger.error(f"Error in coordinate conversion: {e}", exc_info=True)
            # 返回默认值
            return [float(pos[0]), float(pos[2]), -float(pos[1])], [1.0, 0.0, 0.0], 0.0
    
    def _extract_entity_name(self, body_name: str) -> str:
        """
        从 MuJoCo body 名称提取 entityName
        
        直接使用原始 body 名称，无需转换
        (body name 设计满足 Python 变量约束格式)
        
        例如：
            "toys_usda_sphere_body" → "toys_usda_sphere_body"
            "toys_usda_box_body" → "toys_usda_box_body"
        
        Args:
            body_name: MuJoCo body 名称
            
        Returns:
            str: entityName（与 body_name 相同）
        """
        return body_name
    
    def extract_rigid_body_config(self, body_name: str, rb_id: int) -> Dict:
        """
        提取单个刚体的完整配置
        
        Args:
            body_name: MuJoCo body 名称
            rb_id: �体在 scene.json 中的 ID
            
        Returns:
            Dict: 刚体配置（SPH scene.json 格式）
        """
        try:
            body_info = self.extract_body_info(body_name)
            
            # 坐标转换
            translation, rot_axis, rot_angle = self.convert_coordinate_z_to_y(
                body_info['position'],
                body_info['quaternion']
            )
            
            # 生成 entityName
            entity_name = self._extract_entity_name(body_name)
            
            # 获取默认值
            default_rb = self.config.get('default_rigid_body', {})
            
            # 根据 geom_info.is_static 确定 isDynamic
            is_static = body_info['geom_info'].is_static
            is_dynamic = not is_static if is_static else default_rb.get('isDynamic', True)
            
            config = {
                "id": rb_id,
                "entityName": entity_name,
                "geometryFile": body_info['geom_info'].mesh_name,
                "isDynamic": is_dynamic,
                "density": default_rb.get('density', 500),  # 保留作为默认值
                "mass": body_info['mass'],  # 新增：从 MuJoCo 读取的质量
                "translation": translation,
                "rotationAxis": rot_axis,
                "rotationAngle": rot_angle,
                "scale": body_info['geom_info'].scale if body_info['geom_info'].scale else [1, 1, 1],
                "velocity": [0, 0, 0],
                "restitution": default_rb.get('restitution', 0.5),
                "friction": default_rb.get('friction', 0.25),
                "color": [0.5, 0.5, 0.5, 1.0],
                "collisionObjectType": 2,  # 固定为 Box（刚体碰撞关闭时不影响）
                "collisionObjectScale": body_info['geom_info'].scale if body_info['geom_info'].scale else [1.0, 1.0, 1.0],
                "mapInvert": default_rb.get('mapInvert', False),
                "mapThickness": default_rb.get('mapThickness', 0.0),
                "mapResolution": default_rb.get('mapResolution', [20, 20, 20])
            }
            
            body_type = "static" if is_static else "dynamic"
            logger.debug(f"Generated {body_type} rigid body config for '{body_name}': {config}")
            return config
            
        except Exception as e:
            logger.error(f"Error extracting rigid body config for '{body_name}': {e}", exc_info=True)
            raise
    
    def generate_anchor_points(self, main_body_name: str, rb_id: int) -> Optional[Dict]:
        """
        为指定刚体生成 AnchorPoints 配置
        
        使用世界坐标，通过 query_site_pos_and_mat API 读取 SITE 点世界坐标
        
        Args:
            main_body_name: 主刚体名称
            rb_id: 刚体 ID
        
        Returns:
            Dict with anchor points configuration or None if no points found
        """
        # 1. 提取 site 点世界坐标（虚拟锚点）
        site_infos = self.extract_site_positions_for_body(main_body_name)
        site_points = []
        for site in site_infos:
            world_pos_mj = np.array(site['world_position'])
            # 坐标系转换 Z-up → Y-up
            world_pos_sph = self.convert_local_coord_z_to_y(world_pos_mj)
            site_points.append({
                "point_id": site['site_name'],
                "initial_world_pos": world_pos_sph
            })
        
        # 2. 提取 mocap 点全局坐标（牵引点，独立的 mocap body）
        mocap_infos = self.extract_mocap_world_positions_for_body(main_body_name)
        mocap_points = []
        for mocap in mocap_infos:
            world_pos_mj = np.array(mocap['world_position'])
            # 坐标系转换 Z-up → Y-up (和 site 统一处理)
            world_pos_sph = self.convert_local_coord_z_to_y(world_pos_mj)
            mocap_points.append({
                "point_id": mocap['mocap_name'],
                "world_pos": world_pos_sph  # 全局坐标（已转换为 SPH Y-up）
            })
        
        if not site_points and not mocap_points:
            return None
        
        # 弹簧参数现在从 sph_sim_config.json 的 shared_modules.spring_force 直接加载，
        # 不再需要通过场景文件传递，因此移除了这里的配置读取和转发逻辑
        
        return {
            "rigid_body_id": rb_id,
            "object_name": main_body_name,
            "site_points": site_points,
            "mocap_points": mocap_points
        }
    
    
    def generate_scene_json(self, output_path: str = None) -> Dict:
        """
        生成完整的 scene.json RigidBodies 部分
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            Dict: RigidBodies 配置字典
        """
        try:
            # 识别 SPH bodies
            sph_bodies = self.identify_sph_bodies()
            
            if not sph_bodies:
                logger.warning("No SPH bodies found in the model")
                return {"RigidBodies": []}
            
            # 在读取 data 数据前，先调用 mj_forward 更新数据
            self.env.mj_forward()
            
            # 提取每个 body 的配置
            rigid_bodies = []
            for rb_id, body_name in enumerate(sph_bodies, start=1):
                config = self.extract_rigid_body_config(body_name, rb_id)
                rigid_bodies.append(config)
            
            scene_data = {
                "Configuration": self.config.get('scene_template', {}).get('Configuration', {
                    "particleRadius": 0.025,
                    "numberOfStepsPerRenderUpdate": 4,
                    "density0": 1000,
                    "simulationMethod": 4,
                    "gravitation": [0, -9.81, 0],
                    "cflMethod": 1,
                    "cflFactor": 1,
                    "cflMaxTimeStepSize": 0.005,
                    "maxIterations": 100,
                    "maxError": 0.05,
                    "maxIterationsV": 100,
                    "maxErrorV": 0.1,
                    "stiffness": 50000,
                    "exponent": 7,
                    "velocityUpdateMethod": 0,
                    "enableDivergenceSolver": True,
                    "particleAttributes": "velocity",
                    "boundaryHandlingMethod": 2
                }),
                "Materials": self.config.get('scene_template', {}).get('Materials', [
                    {
                        "id": "Fluid",
                        "colorMapType": 1,
                        "surfaceTension": 0.2,
                        "surfaceTensionMethod": 0,
                        "viscosity": 0.01,
                        "viscosityMethod": 1
                    }
                ]),
                "RigidBodies": rigid_bodies
            }
            
            # 保存到文件（如果指定了路径）
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(scene_data, f, indent=4)
                logger.info(f"Scene JSON generated and saved to {output_path}")
            
            logger.info(f"Generated scene with {len(rigid_bodies)} rigid bodies")
            return scene_data
            
        except Exception as e:
            logger.error(f"Error generating scene JSON: {e}", exc_info=True)
            raise
    
    def _init_particle_radius(self) -> None:
        """从 MJCF bound site 的 User 向量读取 particleRadius（user[0]）与传输标签（user[1]）。

        - user[0]: particleRadius（与 Orca ParticleRenderBounds 一致）
        - user[1]: PRBTransmissionMode — 合并到 config particle_render.grpc（与 Editor Transmission Mode 一致）
          0=particle_frame，1=voxel_grid，2=raw_fp32_batch（particle_frame + payload_format raw_fp32）

        必须在 generate_complete_scene() 最开始调用，早于任何依赖 self.particle_radius 的计算。

        仅有 user[0] 的旧 MJCF 保持原有行为；不覆盖 grpc 字段。
        """
        self.particle_render_transmission_tag = None
        try:
            site_dict = self.env.model.get_site_dict()
            if not site_dict:
                return

            bounds_site_name = next(
                (n for n in site_dict if n.endswith("_SPH_PARTICLE_RENDER_BOUNDS")),
                None,
            )
            if bounds_site_name is None:
                return

            user_data = site_dict[bounds_site_name].get("User", [])
            if not user_data or len(user_data) < 1:
                logger.warning(
                    f"Bound site '{bounds_site_name}' has no valid user[0] — "
                    f"falling back to scene_config particleRadius={self.particle_radius}"
                )
                return

            radius = float(user_data[0])
            if radius > 0.0:
                self.particle_radius = radius
                try:
                    self.config["scene_template"]["Configuration"]["particleRadius"] = radius
                except (KeyError, TypeError):
                    pass
                logger.info(
                    f"particleRadius initialized from MJCF site '{bounds_site_name}': {radius}"
                )
            else:
                logger.warning(
                    f"Bound site '{bounds_site_name}' has non-positive user[0]={radius} — "
                    f"keeping scene_config particleRadius={self.particle_radius}"
                )

            if len(user_data) >= 2:
                try:
                    tag = int(round(float(user_data[1])))
                except (TypeError, ValueError):
                    tag = None
                if tag is not None:
                    self.particle_render_transmission_tag = tag
                    pr = self.config.setdefault("particle_render", {})
                    grpc = pr.setdefault("grpc", {})
                    if tag == 0:
                        grpc["transmission_mode"] = "particle_frame"
                    elif tag == 1:
                        grpc["transmission_mode"] = "voxel_grid"
                    elif tag == 2:
                        grpc["transmission_mode"] = "particle_frame"
                        grpc["payload_format"] = "raw_fp32"
                    else:
                        logger.warning(
                            f"Unknown PRBTransmissionMode tag {tag} in site user[1]; "
                            "not merging particle_render.grpc fields"
                        )
                    if tag in (0, 1, 2):
                        logger.info(
                            f"particle_render.grpc merged from MJCF site user[1] "
                            f"(tag={tag}): {dict(grpc)}"
                        )
        except Exception as e:
            logger.warning(f"_init_particle_radius failed: {e}; using default {self.particle_radius}")

    def _find_particle_render_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        查找 _SPH_PARTICLE_RENDER_BOUNDS Box site（由 MjSphFluidBlockComponent 在
        ParticleRenderBoundsComponent 存在时生成），同时返回世界坐标中心和物理半尺寸。

        Site 名称模式: "{bodyName}_SPH_PARTICLE_RENDER_BOUNDS"
        - site pos  → 渲染 entity 世界坐标（Z-up）
        - site size → GetPhysicalHalfExtents() 已经计算好的半尺寸（Z-up，含 wallSizeRatio）

        Returns:
            (center_zup, half_extents_zup) 各为 np.ndarray shape (3,)；
            若未找到 bounds site 返回 None。
        """
        try:
            model = self.env.model
            site_dict = model.get_site_dict()
            if not site_dict:
                logger.warning("No site_dict found, particle_render bounds site not found")
                return None

            bounds_sites = [name for name in site_dict.keys()
                            if name.endswith('_SPH_PARTICLE_RENDER_BOUNDS')]
            if not bounds_sites:
                return None

            site_name = bounds_sites[0]
            if len(bounds_sites) > 1:
                logger.warning(
                    f"Multiple _SPH_PARTICLE_RENDER_BOUNDS sites found: {bounds_sites}. "
                    f"Using '{site_name}'.")

            # --- world position ---
            self.env.mj_forward()
            site_pos_data = self.env.query_site_pos_and_mat([site_name])
            if site_name not in site_pos_data:
                logger.warning(f"Site '{site_name}' not found in query_site_pos_and_mat result")
                return None

            xpos = site_pos_data[site_name]['xpos']
            if isinstance(xpos, np.ndarray):
                center_zup = xpos[:3].copy().astype(float)
            elif isinstance(xpos, (list, tuple)) and len(xpos) >= 3:
                center_zup = np.array(xpos[:3], dtype=float)
            else:
                logger.warning(f"Invalid xpos format for site '{site_name}': {xpos}")
                return None

            # --- half-extents from site size ---
            # Try site_dict first (may expose 'Size' key like get_geom_dict does).
            half_extents_zup = None
            site_info = site_dict.get(site_name, {})
            size_field = site_info.get('Size', None)
            if size_field is not None:
                arr = np.array(size_field, dtype=float).flatten()
                if arr.size >= 3:
                    half_extents_zup = arr[:3].copy()

            # Fall back to model.site_size raw array (standard MuJoCo Python bindings).
            if half_extents_zup is None:
                try:
                    site_id = model.site_name2id(site_name)
                    raw_size = np.array(model.site_size[site_id], dtype=float).flatten()
                    if raw_size.size >= 3:
                        half_extents_zup = raw_size[:3].copy()
                except Exception as inner_e:
                    logger.warning(f"Could not read site_size for '{site_name}': {inner_e}")

            if half_extents_zup is None:
                logger.warning(
                    f"Could not read half-extents for site '{site_name}'; "
                    "falling back to center-only path.")
                return None

            logger.info(
                f"Found particle_render bounds site '{site_name}': "
                f"center(Z-up)={center_zup.tolist()}, half_extents(Z-up)={half_extents_zup.tolist()}")
            return center_zup, half_extents_zup

        except Exception as e:
            logger.error(f"Error finding particle_render bounds site: {e}", exc_info=True)
            return None

    def generate_particle_render_config(self, sph_config: dict) -> Optional[dict]:
        """根据 MJCF bound site 的物理半尺寸计算 particle_render 配置覆盖项。

        将米制的 half_extents 逆推为整数格子数（必须为 2 的幂），分别覆盖
        particle_frame.grid_resolution 和 voxel_grid 的 width/height/depth/origin。

        另外，若在 generate_complete_scene 中已从 MJCF site user[1] 合并了
        ``particle_render.grpc``（例如 raw_fp32），会一并写入返回值，供
        ``generate_orcasph_config`` 写入 orcasph JSON（与仅存在于 scene_config 内存中的
        模板默认值区分开）。

        前置条件：self.particle_radius 已通过 _init_particle_radius() 设置。

        Args:
            sph_config: 从 sph_sim_config.json 加载的完整 SPH 配置字典。

        Returns:
            dict  — 需要覆盖的 particle_render 子字典，可直接 deep-merge 到最终 orcasph_config；
                    若无 bound site 且无 MJCF grpc 覆盖则返回 None。
        """
        import math

        def _round_to_pow2(x: float, lo: int = 32, hi: int = 1024) -> int:
            """将正实数四舍五入到最近的 2 的幂，并 clamp 到 [lo, hi]。"""
            if x <= 0:
                return lo
            p = int(round(math.log2(x)))
            return int(max(lo, min(hi, 2 ** p)))

        override: Optional[dict] = None

        pr_bounds = self._find_particle_render_bounds()
        if pr_bounds is not None:
            center_zup, half_zup = pr_bounds

            # Z-up → Y-up 轴置换：[hx, hy, hz]_zup → [hx, hz, hy]_yup
            half_yup = np.array([half_zup[0], half_zup[2], half_zup[1]], dtype=float)
            center_yup = np.array(self.convert_local_coord_z_to_y(center_zup), dtype=float)

            pr_section = sph_config.get("particle_render", {})

            # ---- particle_frame ----
            pf_cfg = pr_section.get("particle_frame", {})
            vsr_pf = float(pf_cfg.get("voxel_size_ratio", 1.0))
            voxel_pf = self.particle_radius * vsr_pf
            # ---- voxel_grid ----
            vg_cfg = pr_section.get("voxel_grid", {})
            vsr_vg = float(vg_cfg.get("voxel_size_ratio", 1.0))
            voxel_vg = self.particle_radius * 2.0 * vsr_vg

            if voxel_pf <= 0:
                logger.warning(
                    f"Invalid voxel size for particle_frame ({voxel_pf}), skipping geometry override"
                )
            elif voxel_vg <= 0:
                logger.warning(
                    f"Invalid voxel size for voxel_grid ({voxel_vg}), skipping geometry override"
                )
            else:
                res_x = _round_to_pow2(half_yup[0] * 2.0 / voxel_pf)
                res_y = _round_to_pow2(half_yup[1] * 2.0 / voxel_pf)
                res_z = _round_to_pow2(half_yup[2] * 2.0 / voxel_pf)

                vg_w = _round_to_pow2(half_yup[0] * 2.0 / voxel_vg)
                vg_h = _round_to_pow2(half_yup[1] * 2.0 / voxel_vg)
                vg_d = _round_to_pow2(half_yup[2] * 2.0 / voxel_vg)

                override = {
                    "particle_frame": {
                        "grid_resolution": {
                            "x": res_x,
                            "y": res_y,
                            "z": res_z,
                        },
                        "origin": {
                            "x": float(center_yup[0]),
                            "y": float(center_yup[1]),
                            "z": float(center_yup[2]),
                        },
                    },
                    "voxel_grid": {
                        "width": vg_w,
                        "height": vg_h,
                        "depth": vg_d,
                        "origin": {
                            "x": float(center_yup[0]),
                            "y": float(center_yup[1]),
                            "z": float(center_yup[2]),
                        },
                    },
                }

                logger.info(
                    f"particle_render override computed from bound site "
                    f"(particleRadius={self.particle_radius}):\n"
                    f"  particle_frame.grid_resolution = {res_x}×{res_y}×{res_z}, "
                    f"origin={center_yup.tolist()}\n"
                    f"  voxel_grid = {vg_w}×{vg_h}×{vg_d}, origin={center_yup.tolist()}"
                )
        else:
            logger.info(
                "generate_particle_render_config: no bound site found, skipping geometry override"
            )

        mjcf_grpc = (self.config.get("particle_render") or {}).get("grpc")
        if mjcf_grpc:
            if override is None:
                override = {}
            override.setdefault("grpc", {})
            _deep_merge(override["grpc"], dict(mjcf_grpc))
            logger.info(
                "particle_render.grpc from MJCF merged into OrcaSPH override: %s",
                dict(override["grpc"]),
            )

        if override is None:
            return None
        return override

    def generate_complete_scene(self, output_path: str = None, 
                               include_fluid_blocks: bool = True,
                               include_wall: bool = True) -> Dict:
        """
        生成完整的 scene.json 文件（包括 Configuration、Materials、RigidBodies、FluidBlocks）
        
        新增功能：
        1. 为每个主刚体生成 Site 可视化刚体
        2. 为每个主刚体生成 Mocap 可视化刚体
        3. 生成 FixedJoint 约束（Site -> Main）
        4. 生成 RigidBodySpring 约束（Mocap -> Site）
        
        Args:
            output_path: 输出文件路径（可选）
            include_fluid_blocks: 是否包含 FluidBlocks 模板（默认 True）
            include_wall: 是否包含容器墙体（默认 True）
            
        Returns:
            Dict: 完整的场景配置字典
        """
        try:
            # 设置 scene 目录（用于 SPH_MESH 文件的相对路径转换）
            if output_path:
                self.scene_dir = str(Path(output_path).parent.absolute())
                logger.info(f"Scene directory set to: {self.scene_dir}")

            # 在读取 data 数据前，先调用 mj_forward 更新数据
            self.env.mj_forward()

            # 最优先：从 MJCF bound site 读取 particleRadius，确保全程一致。
            # 必须在所有依赖 self.particle_radius 的代码之前调用。
            self._init_particle_radius()

            # 生成主刚体
            main_rigid_bodies = self.generate_scene_json(output_path=None)["RigidBodies"]
            
            # 提前计算 FluidBlocks Y-up 数据（供 wall 尺寸和后续 FluidBlocks 写入共用）
            fluid_blocks_yup = []
            detected_blocks = self.identify_fluid_block_geoms() if include_fluid_blocks else []
            logger.info(f"Detected {len(detected_blocks)} FluidBlocks")
            if detected_blocks:
                for b in detected_blocks:
                    start_yup_raw = np.array(self.convert_local_coord_z_to_y(np.array(b["start"])))
                    end_yup_raw = np.array(self.convert_local_coord_z_to_y(np.array(b["end"])))
                    start_yup = np.minimum(start_yup_raw, end_yup_raw).tolist()
                    end_yup = np.maximum(start_yup_raw, end_yup_raw).tolist()
                    translation_yup = self.convert_local_coord_z_to_y(np.array(b["position"]))
                    fb_entry = {
                        "start_yup": start_yup,
                        "end_yup": end_yup,
                        "translation_yup": translation_yup,
                    }
                    if b.get('wall_mesh_file'):
                        fb_entry['wall_mesh_file'] = b['wall_mesh_file']
                    fluid_blocks_yup.append(fb_entry)
            
            # 准备收集所有刚体
            all_rigid_bodies = []
            
            # 当前 ID 计数器
            current_rb_id = 0
            
            # Wall container (id=0)：有 FluidBlocks 时按 2x/6y/2z 和中心对齐自动计算尺寸
            if include_wall:
                wall_config = self.config.get('wall_rigid_body', {})
                if wall_config:
                    # 优先从 fluid block 的 wall_mesh_file 读取 geometryFile
                    # （由 MjSphFluidBlockComponent 写入 MJCF，通过资产系统管理）
                    wall_geometry_file_from_mjcf = None
                    for fb in fluid_blocks_yup:
                        if fb.get('wall_mesh_file'):
                            wall_geometry_file_from_mjcf = fb['wall_mesh_file']
                            logger.info(
                                f"Wall geometryFile from MJCF mesh asset: '{wall_geometry_file_from_mjcf}'")
                            break

                    if wall_geometry_file_from_mjcf:
                        geometry_file = wall_geometry_file_from_mjcf
                    else:
                        # 回退：转换 scene_config.json 中的 geometryFile 为绝对路径
                        geometry_file = wall_config.get('geometryFile', "../models/UnitBox.obj")
                        geometry_file = self._resolve_geometry_path(geometry_file)
                    
                    wall_translation = wall_config.get('translation', [0, 3.0, 0])
                    wall_scale = wall_config.get('scale', [1.5, 6, 1.5])
                    wall_collision_scale = wall_config.get('collisionObjectScale', [1.5, 6, 1.5])
                    # Use ParticleRenderBoundsComponent Box site when available;
                    # otherwise fall back to the static defaults in wall_rigid_body config.
                    pr_bounds = self._find_particle_render_bounds()
                    if pr_bounds is not None:
                        center_zup, half_zup = pr_bounds
                        center_yup = np.array(self.convert_local_coord_z_to_y(center_zup))
                        # Axis permutation for half-extents: Z-up [hx,hy,hz] → Y-up [hx,hz,hy]
                        half_yup = np.array([half_zup[0], half_zup[2], half_zup[1]], dtype=float)
                        wall_scale       = (half_yup * 2.0).tolist()
                        wall_translation = center_yup.tolist()
                        wall_collision_scale = wall_scale
                        logger.info(
                            f"Wall sized from ParticleRenderBoundsComponent: "
                            f"scale={wall_scale}, translation={wall_translation}")
                    else:
                        logger.info(
                            f"No ParticleRenderBoundsComponent found; "
                            f"using wall_rigid_body config defaults: "
                            f"scale={wall_scale}, translation={wall_translation}")
                    
                    wall_rigid_body = {
                        "id": current_rb_id,
                        "geometryFile": geometry_file,
                        "translation": wall_translation,
                        "rotationAxis": wall_config.get('rotationAxis', [1, 0, 0]),
                        "rotationAngle": wall_config.get('rotationAngle', 0),
                        "scale": wall_scale,
                        "color": wall_config.get('color', [0.1, 0.4, 0.6, 1.0]),
                        "isDynamic": wall_config.get('isDynamic', False),
                        "isWall": wall_config.get('isWall', True),
                        "collisionObjectType": wall_config.get('collisionObjectType', 2),
                        "collisionObjectScale": wall_collision_scale,
                        "invertSDF": wall_config.get('invertSDF', True),
                        "mapInvert": wall_config.get('mapInvert', True),
                        "mapThickness": wall_config.get('mapThickness', 0.0),
                        "mapResolution": wall_config.get('mapResolution', [30, 50, 30])
                    }
                    all_rigid_bodies.append(wall_rigid_body)
                    current_rb_id += 1
            
            # 为每个主刚体添加（不再生成辅助刚体，改用虚拟锚点粒子方案）
            for main_rb in main_rigid_bodies:
                main_body_name = main_rb.get("entityName")
                if not main_body_name:
                    logger.warning(f"Main rigid body missing entityName, skipping: {main_rb}")
                    continue
                
                # 添加主刚体
                main_rb["id"] = current_rb_id
                all_rigid_bodies.append(main_rb)
                current_rb_id += 1
                
                # 提取 site 信息用于后续虚拟锚点配置（不生成辅助刚体）
                site_infos = self.extract_site_positions_for_body(main_body_name)
                if site_infos:
                    logger.debug(f"Found {len(site_infos)} sites for '{main_body_name}' (will be used for anchor particles)")
            
            # 组装完整场景
            scene_template = self.config.get('scene_template', {})
            complete_scene = {
                "Configuration": scene_template.get("Configuration", {}),
                "Materials": scene_template.get("Materials", [])
            }
            
            # 添加刚体
            complete_scene["RigidBodies"] = all_rigid_bodies
            
            # 生成 AnchorPoints
            logger.info("Generating AnchorPoints configuration...")
            anchor_points = []
            for main_rb in main_rigid_bodies:
                main_body_name = main_rb.get("entityName")
                if not main_body_name:
                    continue
                rb_id = main_rb.get("id")
                if rb_id is None:
                    continue
                
                # 跳过静态刚体（不需要锚点）
                if main_rb.get("isDynamic") == False:
                    logger.info(f"  Skipping anchor points for static body '{main_body_name}'")
                    continue
                
                anchor_point = self.generate_anchor_points(main_body_name, rb_id)
                if anchor_point:
                    anchor_points.append(anchor_point)
                    logger.info(f"  Generated {len(anchor_point['site_points'])} site points + "
                               f"{len(anchor_point['mocap_points'])} mocap points for '{main_body_name}'")
            
            # 添加到场景
            if anchor_points:
                complete_scene["AnchorPoints"] = anchor_points
                logger.info(f"Total: {len(anchor_points)} rigid bodies with anchor points")
            
            # 注意：不再生成辅助刚体和约束，虚拟锚点粒子将在运行时通过 PBD 创建
            
            # 添加 FluidBlocks：使用已计算的 fluid_blocks_yup，无重复 identify
            if include_fluid_blocks:
                if fluid_blocks_yup:
                    template_blocks = scene_template.get('FluidBlocks', [{}])
                    template_dense = template_blocks[0].get('denseMode', 0) if template_blocks else 0
                    complete_scene["FluidBlocks"] = [
                        {
                            "denseMode": template_dense,
                            "start": fb["start_yup"],
                            "end": fb["end_yup"],
                            "translation": fb["translation_yup"],
                            "scale": [1.0, 1.0, 1.0],
                        }
                        for fb in fluid_blocks_yup
                    ]
                    logger.info(f"FluidBlocks from {len(fluid_blocks_yup)} SPH_FLUID_BLOCK_GEOM(s)")
                else:
                    complete_scene["FluidBlocks"] = scene_template.get('FluidBlocks', [
                        {
                            "denseMode": 0,
                            "start": [-0.35, -0.35, -0.35],
                            "end": [0.35, 0.35, 0.35],
                            "translation": [0.0, 0.5, 0.0],
                            "scale": [2.0, 1.0, 2.0]
                        }
                    ])
            
            # 保存到文件（如果指定了路径）
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(complete_scene, f, indent=2)
                logger.info(f"Complete scene with anchor system saved to {output_path}")
            
            logger.info(f"Generated complete scene with {len(all_rigid_bodies)} rigid bodies "
                       f"(virtual anchor particles will be created at runtime via PBD)")
            return complete_scene
            
        except Exception as e:
            logger.error(f"Error generating complete scene: {e}", exc_info=True)
            raise


# 便捷函数
def generate_scene_from_env(env, output_path: str = None, config_path: str = None) -> Dict:
    """
    便捷函数：从 OrcaGym 环境生成 scene.json
    
    Args:
        env: OrcaGymLocalEnv 实例
        output_path: 输出文件路径
        config_path: 配置文件路径
        
    Returns:
        Dict: 生成的 scene 配置
    """
    generator = SceneGenerator(env, config_path=config_path)
    return generator.generate_scene_json(output_path)

