#!/usr/bin/env python3
"""
从 XML 文件生成 SPH 场景的命令行工具

从 MuJoCo XML 文件直接解析并生成 SPH scene.json 文件。
无需 OrcaGym 环境，适合调试和分析场景配置。

使用方法:
    python -m envs.fluid.utils.generate_scene_cli <xml_path> <output_json_path> [选项]
    
示例:
    python -m envs.fluid.utils.generate_scene_cli \\
        "/path/to/out.xml" \\
        "scene.json" \\
        --config scene_config.json
"""

import sys
import json
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class XMLSceneGenerator:
    """从 XML 文件直接生成场景"""
    
    def __init__(self, xml_path: str, config: dict):
        self.xml_path = xml_path
        self.config = config
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        logger.info(f"✓ XML 文件已加载: {xml_path}")
    
    def identify_sph_bodies(self) -> List[str]:
        """从 XML 中识别 SPH bodies"""
        sph_bodies = set()
        
        # 查找所有 site 元素
        for site in self.root.findall('.//site'):
            site_name = site.get('name', '')
            if 'SPH_SITE' in site_name:
                # 从 site 名称推断 body 名称
                # 例如: "toys_usda_sphere_body_SPH_SITE_000" -> "toys_usda_sphere_body"
                body_name = site_name.split("_SPH_SITE_")[0]
                sph_bodies.add(body_name)
        
        result = sorted(list(sph_bodies))
        logger.info(f"✓ 识别到 {len(result)} 个 SPH bodies: {result}")
        return result
    
    def get_body_position(self, body_name: str) -> np.ndarray:
        """从 XML 获取 body 的初始位置"""
        for body in self.root.findall('.//body'):
            if body.get('name') == body_name:
                pos_str = body.get('pos', '0 0 0')
                pos = np.array([float(x) for x in pos_str.split()])
                return pos
        return np.array([0, 0, 0])
    
    def get_body_quaternion(self, body_name: str) -> np.ndarray:
        """从 XML 获取 body 的初始旋转（四元数）"""
        for body in self.root.findall('.//body'):
            if body.get('name') == body_name:
                # MuJoCo 使用欧拉角或四元数表示，这里简化处理
                # 返回单位四元数（无旋转）
                return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def get_sph_mesh_file(self, body_name: str) -> str:
        """从 XML 获取 SPH mesh 文件路径"""
        # 查找 SPH_MESH 名称
        sph_mesh_name = f"{body_name}_SPH_MESH"
        
        for mesh in self.root.findall('.//mesh'):
            if mesh.get('name') == sph_mesh_name:
                return mesh.get('file', '')
        
        return ""
    
    def get_mesh_scale(self, mesh_name: str) -> List[float]:
        """从 XML 获取 mesh 的 scale 属性"""
        for mesh in self.root.findall('.//mesh'):
            if mesh.get('name') == mesh_name:
                scale_str = mesh.get('scale', '1.0 1.0 1.0')
                try:
                    scale = [float(x) for x in scale_str.split()]
                    return scale if len(scale) == 3 else [1.0, 1.0, 1.0]
                except ValueError:
                    return [1.0, 1.0, 1.0]
        return [1.0, 1.0, 1.0]
    
    def get_body_mass(self, body_name: str) -> float:
        """从 XML 获取 body 的质量"""
        for body in self.root.findall('.//body'):
            if body.get('name') == body_name:
                inertial = body.find('inertial')
                if inertial is not None:
                    mass = inertial.get('mass', '1.0')
                    return float(mass)
        return 1.0
    
    def generate_scene_json(self, output_path: str = None) -> Dict:
        """生成场景 JSON"""
        sph_bodies = self.identify_sph_bodies()
        rigid_bodies = []
        
        # 添加容器墙体
        wall_config = self.config.get('wall_rigid_body', {})
        if wall_config:
            wall_rb = {
                "id": 0,
                "geometryFile": wall_config.get('geometryFile', "../models/UnitBox.obj"),
                "translation": wall_config.get('translation', [0, 3.0, 0]),
                "rotationAxis": wall_config.get('rotationAxis', [1, 0, 0]),
                "rotationAngle": wall_config.get('rotationAngle', 0),
                "scale": wall_config.get('scale', [1.5, 6, 1.5]),
                "color": wall_config.get('color', [0.1, 0.4, 0.6, 1.0]),
                "isDynamic": wall_config.get('isDynamic', False),
                "isWall": wall_config.get('isWall', True),
                "collisionObjectType": wall_config.get('collisionObjectType', 2),
                "collisionObjectScale": wall_config.get('collisionObjectScale', [1.5, 6, 1.5]),
                "invertSDF": wall_config.get('invertSDF', True),
                "mapInvert": wall_config.get('mapInvert', True),
                "mapThickness": wall_config.get('mapThickness', 0.0),
                "mapResolution": wall_config.get('mapResolution', [30, 50, 30])
            }
            rigid_bodies.append(wall_rb)
        
        # 添加 SPH bodies
        for idx, body_name in enumerate(sph_bodies, start=1):
            pos = self.get_body_position(body_name)
            sph_mesh_name = f"{body_name}_SPH_MESH"
            mesh_file = self.get_sph_mesh_file(body_name)
            scale = self.get_mesh_scale(sph_mesh_name)
            
            # 简单的坐标系转换: Z-up -> Y-up: [x, y, z] -> [x, z, y]
            translation = [float(pos[0]), float(pos[2]), float(pos[1])]
            
            rb = {
                "id": idx,
                "entityName": body_name,
                "geometryFile": mesh_file if mesh_file else "../models/box_small.obj",
                "isDynamic": True,
                "density": self.config.get('default_rigid_body', {}).get('density', 500),
                "translation": translation,
                "rotationAxis": [1.0, 0.0, 0.0],
                "rotationAngle": 0.0,
                "scale": scale,
                "velocity": [0, 0, 0],
                "collisionObjectType": 2,  # 默认 box
                "restitution": self.config.get('default_rigid_body', {}).get('restitution', 0.5),
                "friction": self.config.get('default_rigid_body', {}).get('friction', 0.25),
                "mapInvert": self.config.get('default_rigid_body', {}).get('mapInvert', False),
                "mapThickness": self.config.get('default_rigid_body', {}).get('mapThickness', 0.0),
                "mapResolution": self.config.get('default_rigid_body', {}).get('mapResolution', [20, 20, 20])
            }
            rigid_bodies.append(rb)
        
        scene_data = {
            "Configuration": self.config.get('scene_template', {}).get('Configuration', {}),
            "Materials": self.config.get('scene_template', {}).get('Materials', []),
            "RigidBodies": rigid_bodies,
            "FluidBlocks": self.config.get('scene_template', {}).get('FluidBlocks', [])
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
            logger.info(f"✓ 场景已生成: {output_path}")
        
        return scene_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从 XML 文件生成 SPH 场景',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m envs.fluid.utils.generate_scene_cli \\
    "/path/to/out.xml" "scene.json" \\
    --config scene_config.json
        """
    )
    
    parser.add_argument('xml_path', help='MuJoCo XML 文件路径')
    parser.add_argument('output_path', help='输出 scene.json 文件路径')
    parser.add_argument('--config', default='scene_config.json', help='配置文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 验证输入
        xml_path = Path(args.xml_path)
        if not xml_path.exists():
            logger.error(f"XML 文件不存在: {xml_path}")
            return False
        
        # 尝试从多个位置查找配置文件
        config_path = Path(args.config)
        if not config_path.exists():
            # 尝试从 examples/fluid/ 目录查找
            examples_config = Path(__file__).parent.parent.parent.parent / "examples" / "fluid" / args.config
            if examples_config.exists():
                config_path = examples_config
            else:
                logger.error(f"配置文件不存在: {args.config}")
                return False
        
        logger.info(f"📁 XML 文件: {xml_path}")
        logger.info(f"⚙️  配置文件: {config_path}")
        logger.info(f"📝 输出文件: {args.output_path}")
        
        # 加载配置
        with open(config_path) as f:
            config = json.load(f)
        
        # 从 XML 文件生成场景
        generator = XMLSceneGenerator(str(xml_path), config)
        scene_data = generator.generate_scene_json(args.output_path)
        
        # 统计和验证
        logger.info("\n" + "=" * 80)
        logger.info("✅ 场景生成成功")
        logger.info("=" * 80)
        
        rigid_bodies = scene_data.get('RigidBodies', [])
        logger.info(f"\n📊 RigidBodies: {len(rigid_bodies)} 个")
        for rb in rigid_bodies:
            entity = rb.get('entityName', '(Wall)')
            logger.info(f"  - id={rb['id']}, entity={entity}, file={rb.get('geometryFile')}")
        
        logger.info(f"\n✨ 完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 错误: {e}", exc_info=args.verbose)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

