import open3d as o3d
import numpy as np
import teaserpp_python

# 读取点云数据
source = o3d.io.read_point_cloud("source_point_cloud.pcd")
target = o3d.io.read_point_cloud("target_point_cloud.pcd")

# 下采样点云数据
voxel_size = 0.05
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# 计算 FPFH 特征
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# 使用 TEASER++ 进行配准
solver = teaserpp_python.RobustRegistrationSolver()
solver.solve(source_down.points, target_down.points)

solution = solver.getSolution()
transformation = np.eye(4)
transformation[:3, :3] = solution.rotation
transformation[:3, 3] = solution.translation

print("Transformation matrix:")
print(transformation)

# 可视化配准结果
source.transform(transformation)
o3d.visualization.draw_geometries([source, target])
