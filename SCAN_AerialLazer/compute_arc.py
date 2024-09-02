import numpy as np
import open3d as o3d
import trimesh

# mesh = trimesh.load("pointcloud/cloudyuanzhu1.ply")
#
# Tform = mesh.apply_obb()
#
# mesh.export("pointcloud/obb_yuanzhu.ply", encoding="ascii")

# 示例：加载点云（这里你需要替换为实际的点云数据文件路径）

pcd = o3d.io.read_point_cloud("pointcloud/cloudyuanzhu1.ply")

aix = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

pcd = o3d.io.read_point_cloud("pointcloud/obb_yuanzhu.ply")


o3d.visualization.draw_geometries([pcd, aix])

# 1. 估计点云的法线
# 设置搜索半径，并使用 KDTree 搜索点云的局部邻域来估计法线
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 可视化估计的法线
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# 2. 选择一个代表性的法线
# 对所有估计的法线进行平均，得到一个总体法线
normals = np.asarray(pcd.normals)
average_normal = np.mean(normals, axis=0)
average_normal /= np.linalg.norm(average_normal)  # 单位化

# 3. 定义平面
# 选择点云中的一点作为平面上的一点
point = np.mean(np.asarray(pcd.points), axis=0)  # 使用点云的质心作为平面上的点

# 计算 d 值
d = -point.dot(average_normal)

# 通过点法式计算每个点到平面的距离
points = np.asarray(pcd.points)
distances = points.dot(average_normal) + d

# 设置截取距离的阈值，只有距离小于阈值的点才被保留
threshold = 0.01
indices = np.where(np.abs(distances) < threshold)[0]

# 4. 截取点云并生成圆弧
arc_pcd = pcd.select_by_index(indices)

# 可视化截取后的圆弧
o3d.visualization.draw_geometries([arc_pcd])

# 5. 保存截取后的圆弧点云
o3d.io.write_point_cloud("pointcloud/arc_segment.ply", arc_pcd)
