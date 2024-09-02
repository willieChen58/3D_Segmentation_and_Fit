import open3d as o3d
import numpy as np
import pyrealsense2 as rs

def triangulation(point_2d, baseline, focal_length, sensor_height, pixel_height):
    sensor_y = (point_2d - pixel_height / 2) * sensor_height / pixel_height
    depth = (baseline * focal_length) / sensor_y
    return depth

# 读取点云数据
pcd = o3d.io.read_point_cloud("pointcloud/jiaosai.ply")

# 应用统计滤波去除离群点
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
inlier_cloud = pcd.select_by_index(ind)

# 计算点云的中心（质心）
center = np.mean(np.asarray(inlier_cloud.points), axis=0)
print(center[0], center[1], center[2])
print("Point Cloud Center (Camera Coordinate System):", center)

# 使用PCA计算点云的主要方向
points = np.asarray(inlier_cloud.points)
covariance = np.cov(points.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# 主要方向是特征值最大的特征向量
principal_direction = eigenvectors[:, np.argmax(eigenvalues)]

# 使用主要方向来定义旋转
z_axis = principal_direction
x_axis = np.array([1, 0, 0])
y_axis = np.cross(z_axis, x_axis)
rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

# 打印旋转矩阵
print("Rotation Matrix:")
print(rotation_matrix)

# 将旋转应用于初始位姿
pose = np.eye(4)
pose[:3, :3] = rotation_matrix
pose[:3, 3] = center

# 打印估计的位姿
print("Estimated Pose:")
print(pose)

# 可视化去噪后的点云及其坐标系
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
# inlier_cloud.translate(-center)
o3d.visualization.draw_geometries([inlier_cloud, mesh_frame])
