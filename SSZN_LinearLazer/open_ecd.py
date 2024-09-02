import open3d as o3d
import numpy as np


def read_custom_ecd(file_path):
    points = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            # 假设每行包含一个点的 x, y, z 坐标，逗号分隔
            parts = line.strip().split(',')
            if len(parts) == 3:
                x, y, z = map(float, parts)
                points.append([x, y, z])

    # 将点云数据转换为 numpy 数组
    points = np.array(points)
    return points


# 读取自定义 .ecd 点云文件
file_path = "sr7400/yp_a/bottom.ecd"
points = read_custom_ecd(file_path)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化点云数据
o3d.visualization.draw_geometries([pcd])
