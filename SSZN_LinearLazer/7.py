import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor

# 假设你有一个阈值，比如 z_threshold = 0.5
z_threshold1 = 3.5

z_threshold2 = 3.8


def find_radius_top(points, center):

    min_distance = 1000
    min_pair = None
    for i in range(len(points)):
        if abs(points[i][1] - center[1]) < 1e-1:
            dist = abs(points[i][0] - center[0])
            if dist < min_distance:
                min_distance = dist
                min_pair = (points[i], center)
    # 返回直径和端点
    return min_distance, min_pair

def remove_low_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]
    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标高于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    return kept_points, high_z_indices


def remove_high_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]

    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标不低于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    # 返回过滤后的点和被移除的点的索引（如果需要的话）
    return kept_points, low_z_indices


def fit_circle_ransac(points, threshold=0.01, max_trials=1000):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def dist_func(c):
        Ri = calc_R(*c)
        return np.abs(Ri - Ri.mean())

    def fit_circle_to_points(points):
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        b = np.sum(points ** 2, axis=1)
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        center_x = c[0] / 2
        center_y = c[1] / 2
        radius = np.sqrt(c[2] + center_x ** 2 + center_y ** 2)
        return (center_x, center_y), radius

    ransac = RANSACRegressor(base_estimator=None, min_samples=3, residual_threshold=threshold, max_trials=max_trials)
    ransac.fit(points[:, :2], np.zeros(points.shape[0]))
    inlier_mask = ransac.inlier_mask_

    inlier_points = points[inlier_mask]
    center, radius = fit_circle_to_points(inlier_points)
    radius = calc_R(*center).mean()
    return center, radius, inlier_mask


def main():
    # 读取点云
    pcd = o3d.io.read_point_cloud("pointcloud/11.ply")

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    points = np.asarray(inlier_cloud.points)


    # 调用函数来过滤点云
    # 现在 filtered_points 只包含z坐标不低于阈值的点
    filtered_points, high_z_indices = remove_low_z_points(points, 3.5)
    inlier_cloud = inlier_cloud.select_by_index(high_z_indices)  # 在已经过滤的点云中保留z坐标高于阈值的点

    points2 = np.asarray(inlier_cloud.points)

    filtered_points, low_z_indices = remove_high_z_points(points2, 3.8)

    inlier_cloud_low = inlier_cloud.select_by_index(low_z_indices)

    # 体素下采样降噪
    inlier_cloud_low = inlier_cloud_low.voxel_down_sample(voxel_size=0.02)
    # o3d.io.write_point_cloud('./pointcloud/22.ply', inlier_cloud_low, write_ascii=True, compressed=False, print_progress=False)

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    print("质心为：", centroid[0], centroid[1], centroid[2])

    # 提取轮廓
    hull = inlier_cloud_low.compute_convex_hull()
    contour_points = np.asarray(hull[0].vertices)

    # 使用RANSAC拟合内圆和外圆
    center, radius, inlier_mask = fit_circle_ransac(contour_points)

    print(center, radius)

    # 可视化结果
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    inlier_cloud_low.paint_uniform_color([0.7, 0.7, 0.7])

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    center_sphere.translate([*center, centroid[2]])
    center_sphere.paint_uniform_color([0, 0, 1])

    inlier_cloud2 = o3d.geometry.PointCloud()
    inlier_cloud2.points = o3d.utility.Vector3dVector(contour_points[inlier_mask])
    inlier_cloud2.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud_low, mesh, inlier_cloud2])



if __name__ == "__main__":
    main()
