"""
SPH-MuJoCo 配置生成器

从 MuJoCo 模型动态生成 sph_mujoco_config.json 的 rigid_bodies 部分。
自动识别 SPH_SITE 和 SPH_MOCAP_SITE，生成连接点配置。
"""

import logging
from typing import Dict, List
import re

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[ConfigGenerator] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


class ConfigGenerator:
    """SPH-MuJoCo 配置生成器"""
    
    def __init__(self, env):
        """
        初始化配置生成器
        
        Args:
            env: OrcaGymLocalEnv 实例
        """
        import sys
        print("[PRINT-DEBUG] ConfigGenerator.__init__() - START", file=sys.stderr, flush=True)
        self.env = env
        self.model = env.model
        print("[PRINT-DEBUG] ConfigGenerator.__init__() - END", file=sys.stderr, flush=True)
    
    def _resolve_body_name_from_site(self, site_data: dict) -> str:
        """
        通过 site 的 BodyID 解析真实的 MuJoCo body 名称。

        site_dict 中每条记录包含 'BodyID' 字段（MuJoCo 原生 body id），
        利用 model.body_id2name() 可靠地拿到 body 名称，
        避免依赖 site 名称前缀与 body 名称一致的假设。
        """
        body_id = site_data.get('BodyID')
        if body_id is None:
            return ""
        try:
            return self.model.body_id2name(int(body_id)) or ""
        except Exception:
            return ""

    def _resolve_internal_prefix(self, body_name: str) -> str:
        """
        解析 body 对应的 SPH site/mocap 命名前缀（内部 ID 前缀）。

        OrcaStudio 导出的 MJCF 中，body 名可能是描述性的（如
        "Group_Interactive_Cup_01_fluid_02_bodyjoint"），而 site/mocap 名
        使用内部 ID 前缀（如 "[11519992300482]_bodyjoint_SPH_SITE_000"）。
        本方法通过 site 的 BodyID 反查内部 ID 前缀，保证前缀匹配可靠。
        """
        try:
            site_dict = self.model.get_site_dict()
            if not site_dict:
                return body_name
            try:
                target_body_id = int(self.model.body_name2id(body_name))
            except Exception:
                return body_name
            for site_name, site_data in site_dict.items():
                if "_SPH_SITE_" not in site_name:
                    continue
                site_body_id = site_data.get('BodyID')
                if site_body_id is not None and int(site_body_id) == target_body_id:
                    return site_name.split("_SPH_SITE_")[0]
            return body_name
        except Exception:
            return body_name

    def identify_sph_bodies(self) -> List[str]:
        """
        识别所有带有 SPH_SITE 的 body

        优先通过 site 的 BodyID 解析真实 body 名称（兼容 site 名称前缀
        与 body 名称不一致的场景，例如 OrcaStudio 导出的 XML 中
        site 名使用内部 ID 前缀如 "[11519992300482]_bodyjoint_SPH_SITE_000"，
        而 body 名为 "Group_Interactive_Cup_01_fluid_02_bodyjoint"）。
        若 BodyID 不可用，则回退到从 site 名称推断。

        Returns:
            List[str]: body 名称列表
        """
        sph_bodies = set()

        try:
            body_names = self.model.get_body_names()

            # 使用 OrcaGymModel API 获取所有 site 的字典
            site_dict = self.model.get_site_dict()

            # 遍历所有 site 名称，识别带有 SPH_SITE 标记的 body
            for site_name, site_data in site_dict.items():
                if "_SPH_SITE_" not in site_name:
                    continue

                # 优先：通过 BodyID 解析真实 body 名称
                body_name = self._resolve_body_name_from_site(site_data)

                # 回退：从 site 名称推断 body 名称
                if not body_name:
                    body_name = site_name.split("_SPH_SITE_")[0]

                # 验证 body 是否存在
                if body_name and body_name in body_names:
                    sph_bodies.add(body_name)
                    logger.debug(f"Identified SPH body: {body_name} (from site '{site_name}')")

            result = sorted(list(sph_bodies))
            logger.info(f"Identified {len(result)} SPH bodies: {result}")
            return result

        except Exception as e:
            logger.error(f"Error identifying SPH bodies: {e}", exc_info=True)
            return []
    
    def extract_sites_for_body(self, body_name: str) -> Dict[str, List[str]]:
        """
        为指定 body 提取所有 SPH_SITE 和 SPH_MOCAP_SITE
        
        Args:
            body_name: MuJoCo body 名称
        
        Returns:
            Dict with keys:
                - 'sph_sites': List of SPH_SITE names (sorted by index)
                - 'mocap_sites': List of SPH_MOCAP_SITE names (sorted by index)
        """
        sph_sites = []
        mocap_sites = []
        
        try:
            # 使用 OrcaGymModel API 获取所有 site 的字典
            site_dict = self.model.get_site_dict()

            # site 名前缀可能与 body 名不一致（OrcaStudio 导出场景），用内部 ID 前缀匹配
            internal_prefix = self._resolve_internal_prefix(body_name)

            # 遍历所有 site 名称
            for site_name in site_dict.keys():
                # 检查是否是该 body 的 SPH_SITE
                if site_name.startswith(f"{internal_prefix}_SPH_SITE_"):
                    sph_sites.append(site_name)

                # 检查是否是该 body 的 SPH_MOCAP_SITE
                if site_name.startswith(f"{internal_prefix}_SPH_MOCAP_SITE_"):
                    mocap_sites.append(site_name)
            
            # 按索引排序
            def extract_index(name: str) -> int:
                """从 site 名称提取索引"""
                match = re.search(r'(\d+)$', name)
                return int(match.group(1)) if match else 999
            
            sph_sites.sort(key=extract_index)
            mocap_sites.sort(key=extract_index)
            
            logger.debug(f"Body '{body_name}': {len(sph_sites)} SPH_SITE, {len(mocap_sites)} MOCAP_SITE")
            
            return {
                'sph_sites': sph_sites,
                'mocap_sites': mocap_sites
            }
            
        except Exception as e:
            logger.error(f"Error extracting sites for body '{body_name}': {e}")
            return {'sph_sites': [], 'mocap_sites': []}
    
    def generate_connection_points(self, body_name: str, sph_sites: List[str], mocap_sites: List[str]) -> List[Dict]:
        """
        生成 connection_points 配置
        
        Args:
            body_name: MuJoCo body 名称
            sph_sites: SPH_SITE 名称列表（已排序）
            mocap_sites: SPH_MOCAP_SITE 名称列表（已排序）
        
        Returns:
            List[Dict]: connection_points 配置列表
        """
        connection_points = []
        
        # 确保两个列表长度一致
        min_len = min(len(sph_sites), len(mocap_sites))
        
        if len(sph_sites) != len(mocap_sites):
            logger.warning(f"Body '{body_name}': SPH_SITE count ({len(sph_sites)}) != MOCAP_SITE count ({len(mocap_sites)})")
        
        for i in range(min_len):
            site_name = sph_sites[i]
            mocap_site_name = mocap_sites[i]

            # 从 site 名称提取内部 ID 前缀，用于构造 mocap body 名称
            internal_prefix = site_name.split("_SPH_SITE_")[0]

            # 从 mocap_site_name 提取索引
            match = re.search(r'SPH_MOCAP_SITE_(\d+)$', mocap_site_name)
            if match:
                index = int(match.group(1))
            else:
                match = re.search(r'SPH_SITE_(\d+)$', site_name)
                index = int(match.group(1)) if match else i

            # 构造 mocap BODY 名称（使用内部 ID 前缀，与 XML 中实际 mocap body 名一致）
            mocap_body_name = f"{internal_prefix}_SPH_MOCAP_{index:03d}"
            
            connection_points.append({
                "point_id": site_name,  # 使用实际 site 名称
                "site_name": site_name,
                "mocap_name": mocap_body_name  # 使用 mocap body 名称
            })
        
        return connection_points
    
    def generate_rigid_bodies(self) -> List[Dict]:
        """
        从 MuJoCo 模型生成完整的 rigid_bodies 配置
        
        Returns:
            List[Dict]: rigid_bodies 配置列表
        """
        rigid_bodies = []
        
        # 识别所有 SPH bodies
        sph_bodies = self.identify_sph_bodies()
        
        if not sph_bodies:
            logger.warning("No SPH bodies found in the model")
            return []
        
        # 为每个 body 生成配置
        for body_name in sph_bodies:
            # 提取 sites
            sites_info = self.extract_sites_for_body(body_name)
            sph_sites = sites_info['sph_sites']
            mocap_sites = sites_info['mocap_sites']
            
            if not sph_sites:
                logger.warning(f"Body '{body_name}' has no SPH_SITE, skipping")
                continue
            
            # 生成 connection_points
            connection_points = self.generate_connection_points(body_name, sph_sites, mocap_sites)
            
            if not connection_points:
                logger.warning(f"Body '{body_name}' has no connection points, skipping")
                continue
            
            # 生成 rigid body 配置
            # object_id 使用 body 名称（或可以提取简化名称，这里直接使用 body_name）
            rigid_body = {
                "object_id": body_name,
                "mujoco_body": body_name,
                "coupling_mode": "multipoint_connect",
                "connection_points": connection_points,
                "spring_params": {
                    "stiffness": 5000.0,
                    "damping": 100.0
                }
            }
            
            rigid_bodies.append(rigid_body)
            logger.info(f"Generated config for body '{body_name}': {len(connection_points)} connection points")
        
        logger.info(f"Generated {len(rigid_bodies)} rigid body configurations")
        return rigid_bodies

