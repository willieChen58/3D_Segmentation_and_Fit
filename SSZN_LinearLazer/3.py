import numpy as np
import open3d as o3d

# 读取点云数据
pcd = o3d.io.read_point_cloud("pointcloud/1.ply")

# 应用统计滤波去除离群点
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
inlier_cloud = pcd.select_by_index(ind)

points = np.asarray(inlier_cloud.points)
# 假设 pcd.points 是一个 NumPy 数组，其形状为 (N, 3)，其中 N 是点的数量
# 并且每行包含 (x, y, z) 坐标
def remove_low_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]

    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标不低于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    # 返回过滤后的点和被移除的点的索引（如果需要的话）
    return kept_points, high_z_indices


# 假设你有一个阈值，比如 z_threshold = 0.5
z_threshold = 30

# 调用函数来过滤点云
# 现在 filtered_points 只包含z坐标不低于阈值的点
filtered_points, high_z_indices = remove_low_z_points(points, z_threshold)

inlier_cloud_high = inlier_cloud.select_by_index(high_z_indices)

# 计算点云的中心（质心）
center = np.mean(np.asarray(inlier_cloud_high.points), axis=0)
print(center[0], center[1], center[2])
print("Point Cloud Center (Camera Coordinate System):", center)

# 可视化去噪后的点云及其坐标系
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
inlier_cloud_high.translate(-center)
o3d.visualization.draw_geometries([inlier_cloud_high, mesh_frame])
