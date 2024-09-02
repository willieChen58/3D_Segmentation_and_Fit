import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def extract_contour(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_vertices = np.asarray(hull.vertices)
    contour_points = np.asarray(pcd.points)[hull_vertices]
    return contour_points

def fit_circle_ransac(points, threshold=0.01, max_trials=1000):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 1] - yc) ** 2 + (points[:, 2] - xc) ** 2)

    def dist_func(c):
        Ri = calc_R(*c)
        return np.abs(Ri - Ri.mean())

    def fit_circle_to_points(points):
        A = np.c_[points[:, 1], points[:, 2], np.ones(points.shape[0])]
        b = np.sum(points[:, 1:] ** 2, axis=1)
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        center_y = c[0] / 2
        center_z = c[1] / 2
        radius = np.sqrt(c[2] + center_y ** 2 + center_z ** 2)
        return (center_y, center_z), radius

    ransac = RANSACRegressor(base_estimator=None, min_samples=3, residual_threshold=threshold, max_trials=max_trials)
    ransac.fit(points[:, 1:], np.zeros(points.shape[0]))
    inlier_mask = ransac.inlier_mask_

    inlier_points = points[inlier_mask]
    center, radius = fit_circle_to_points(inlier_points)
    radius = calc_R(*center).mean()
    return center, radius, inlier_mask

def main():
    file_path = 'pointcloud/obb_bottle_top.ply'  # 替换为你的点云文件路径
    pcd = load_point_cloud(file_path)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)

    # 体素下采样降噪
    inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.02)

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    print("质心为：", centroid[0], centroid[1], centroid[2])

    # 提取轮廓
    # 提取轮廓
    hull = inlier_cloud.compute_convex_hull()
    contour_points = np.asarray(hull[0].vertices)

    # 使用RANSAC拟合内圆和外圆
    center, radius, inlier_mask = fit_circle_ransac(contour_points)
    print(f"圆心 (Y, Z): {center}, 半径: {radius}")

    # 可视化点云和拟合的圆
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        y = center[0] + radius * np.cos(angle)
        z = center[1] + radius * np.sin(angle)
        circle_points.append([centroid[0], y, z])

    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(circle_points)
    circle_pcd.paint_uniform_color([1, 0, 0])

    # 可视化结果
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    center_sphere.translate([centroid[0], *center])
    center_sphere.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd, mesh, circle_pcd])

if __name__ == "__main__":
    main()
