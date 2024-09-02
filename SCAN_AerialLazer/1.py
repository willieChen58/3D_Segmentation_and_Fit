import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares


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

def remove_high_x_points(points, x_threshold):
    # 计算z坐标高于阈值的点的索引
    high_x_indices = np.where(points[:, 0] > x_threshold)[0]
    low_x_indices = np.where(points[:, 0] < x_threshold)[0]

    # 保留z坐标高于阈值的点
    kept_points = np.delete(points, low_x_indices, axis=0)

    return kept_points, low_x_indices

def remove_high_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]
    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标不低于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    # 返回过滤后的点和被移除的点的索引（如果需要的话）
    return kept_points, low_z_indices

def extract_contour(points):
    hull = ConvexHull(points[:, :2])
    contour_points = points[hull.vertices]

    return contour_points


def fit_circle(points):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)[:2]
    center, _ = least_squares(f, center_estimate).x, f(center_estimate)
    radius = calc_R(*center).mean()
    return center, radius


def find_inner_outer_circles(contour_points, threshold=0.01):
    distances = np.linalg.norm(contour_points[:, :2], axis=1)
    median_distance = np.median(distances)
    print(f"Median Distance: {median_distance}")

    inner_circle_points = contour_points[distances < median_distance]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inner_circle_points)
    o3d.visualization.draw_geometries([pcd])


    outer_circle_points = contour_points[distances >= median_distance]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(outer_circle_points)
    o3d.visualization.draw_geometries([pcd])

    inner_center, inner_radius = fit_circle(inner_circle_points)
    outer_center, outer_radius = fit_circle(outer_circle_points)

    return (inner_center, inner_radius), (outer_center, outer_radius)


def main():
    # 读取点云
    # pcd = o3d.io.read_point_cloud("pointcloud/scan.ply")
    #
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # inlier_cloud = pcd.select_by_index(ind)
    # points = np.asarray(inlier_cloud.points)
    #
    # # 调用函数来过滤点云
    # # 现在 filtered_points 只包含z坐标不低于阈值的点
    # filtered_points, low_z_indices = remove_high_z_points(points, 230)
    #
    # inlier_cloud = inlier_cloud.select_by_index(low_z_indices)  # 在已经过滤的点云中保留z坐标高于阈值的点
    #
    # o3d.io.write_point_cloud('./pointcloud/11.ply', inlier_cloud, write_ascii=True, compressed=False,
    #                          print_progress=False)

    inlier_cloud = o3d.io.read_point_cloud("pointcloud/22.ply")
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([inlier_cloud, mesh])

    # 体素下采样降噪
    inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.02)

    points = np.asarray(inlier_cloud.points)

    # filtered_points, low_x_indices = remove_high_x_points(points, -0.88697687)
    # inlier_cloud = inlier_cloud.select_by_index(low_x_indices)
    # points = np.asarray(inlier_cloud.points)

    print(np.mean(points, axis=0))

    # 提取轮廓
    contour_points = extract_contour(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contour_points)
    o3d.visualization.draw_geometries([pcd])

    # 找到内圆和外圆
    (inner_center, inner_radius), (outer_center, outer_radius) = find_inner_outer_circles(contour_points)

    # 计算圆环的中心点
    ring_center = np.mean([inner_center, outer_center], axis=0)

    print(f"Ring Center: {ring_center}")
    print(f"Inner Circle Center: {inner_center}, Radius: {inner_radius}")
    print(f"Outer Circle Center: {outer_center}, Radius: {outer_radius}")

    # 可视化结果
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 0])

    inner_circle = o3d.geometry.TriangleMesh.create_sphere(radius=5.735)
    inner_circle.translate([*inner_center, 30.81484161])
    inner_circle.paint_uniform_color([1, 0, 0])

    outer_circle = o3d.geometry.TriangleMesh.create_sphere(radius=9.3105)
    outer_circle.translate([*outer_center, 0.03519804])
    outer_circle.paint_uniform_color([0, 1, 0])

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
    center_sphere.translate([*ring_center, 0])
    center_sphere.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([inlier_cloud, mesh, outer_circle])


if __name__ == "__main__":
    main()

