import open3d as o3d
import numpy as np

x_threshold = 70
z_threshold = 30
pcd_path = "pointcloud/3.ply"

def remove_high_x_points(points, x_threshold):
    # 计算x坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 0] > x_threshold)[0]
    low_z_indices = np.where(points[:, 0] < x_threshold)[0]

    # 保留x坐标低于阈值的点
    kept_points = np.delete(points, high_z_indices, axis=0)

    return kept_points, low_z_indices

def remove_low_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]
    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标高于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    return kept_points, high_z_indices


def compute_Center(pcd_path, x_threshold, z_threshold, CAL_TOP = False, CAL_PLATFORM = False):
    # 读取点云数据
    pcd = o3d.io.read_point_cloud(pcd_path)

    # 应用统计滤波去除离群点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    points = np.asarray(inlier_cloud.points)

    if CAL_TOP:
        # 现在 filtered_points 只包含z坐标不低于阈值的点
        filtered_points, high_z_indices = remove_low_z_points(points, z_threshold)
        inlier_cloud = inlier_cloud.select_by_index(high_z_indices)  # 在已经过滤的点云中保留z坐标高于阈值的点

        # 计算点云的中心（质心）
        center = np.mean(np.asarray(inlier_cloud.points), axis=0)
        print("TopCenter:", center)
    if CAL_PLATFORM:
        # 现在 filtered_points 只包含z坐标不低于阈值的点
        filtered_points, low_z_indices = remove_high_x_points(points, x_threshold)
        inlier_cloud = inlier_cloud.select_by_index(low_z_indices)  # 在已经过滤的点云中保留z坐标高于阈值的点

        # 计算点云的中心（质心）
        center = np.mean(np.asarray(inlier_cloud.points), axis=0)
        print("Platform_Center:", center)

    # 可视化去噪后的点云及其坐标系
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
    # inlier_cloud.translate(-center)
    # o3d.visualization.draw_geometries([inlier_cloud, mesh_frame])

    return center

def visualize_height(pcd_path, point, center):

    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([point, center]),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[255, 0, 0]])  # 红色的线

    pcd = o3d.io.read_point_cloud(pcd_path)
    # 应用统计滤波去除离群点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries([inlier_cloud, line_set], window_name="Point Cloud with Diameter")

    return 0

if __name__ == "__main__":

    center1 = compute_Center(pcd_path, x_threshold, z_threshold, CAL_PLATFORM=True)
    center2 = compute_Center(pcd_path, x_threshold, z_threshold, CAL_TOP=True)

    print("Bottle Height:", center2[2]-center1[2])

    # 可视化点云和高度
    point2 = center2.copy()
    point2[2] = center1[2]

    visualize_height(point2, center2)
