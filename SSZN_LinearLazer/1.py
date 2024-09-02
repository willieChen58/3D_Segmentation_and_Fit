import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设你已经有一个点云文件，如.pcd或.xyz
pcd = o3d.io.read_point_cloud("../scan/pointcloud/bottle_body.ply")
# 计算点云的中心
print("center:")
center = np.mean(np.asarray(pcd.points), axis=0)
print(center)  # [ -1.88565864  -1.9103358  -10.75015294]
#
# # 假设我们知道点云的初始姿态，这里只是一个例子
initial_pose = np.eye(4)
initial_pose[:3, 3] = center

# 打印初始位姿
print("Initial Pose:")
print(initial_pose)

# 计算点云的主要方向
covariance = np.cov(np.asarray(pcd.points).T)
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# 使用主要方向来定义旋转
rotation_matrix = eigenvectors
r = R.from_matrix(rotation_matrix)
quaternion = r.as_quat()

# 打印旋转矩阵和四元数
print("Rotation Matrix:")
print(rotation_matrix)
print("Quaternion:")
print(quaternion)

# 将旋转应用于初始位姿
pose = np.eye(4)
pose[:3, :3] = rotation_matrix
pose[:3, 3] = center

# 打印最终位姿
print("Final Pose:")
print(pose)

# 可视化点云及其坐标系
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
pcd.translate(-center)
o3d.visualization.draw_geometries([pcd, mesh_frame])
