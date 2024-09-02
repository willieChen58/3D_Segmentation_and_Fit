import open3d as o3d
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

def load_point_cloud(file_path):
    # 加载点云数据
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def extract_contour(points):
    # 将点云转换为二维点
    points_2d = points[:, 1:]

    # 使用OpenCV的凸包函数提取轮廓
    hull = cv2.convexHull(points_2d.astype(np.float32))

    # 转换回Numpy数组格式
    contour_points = hull[:, 0, :]     # [ 9.12429  -0.641734]...

    return contour_points


def fit_circle_2d(points):
    # 计算中心点和半径的初始估计
    center_estimate = np.mean(points, axis=0)
    print("质心估计为：", center_estimate)
    radii_estimate = np.linalg.norm(points - center_estimate, axis=1).mean()

    # 使用最小二乘法拟合圆
    def residuals(params, points):
        cx, cy, r = params
        return np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) - r

    result = least_squares(residuals, x0=[center_estimate[0], center_estimate[1], radii_estimate], args=(points,))

    # 提取拟合圆的参数
    cx, cy, r = result.x
    return np.array([cx, cy]), r


def main():
    file_path = 'pointcloud/obb_bottle_top.ply'  # 替换为你的点云文件路径
    pcd = load_point_cloud(file_path)

    # 获取点云的点坐标
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    print("质心为：", centroid[0], centroid[1], centroid[2])

    # 提取点云轮廓
    contour_points = extract_contour(points)

    # 拟合二维圆
    center, radius = fit_circle_2d(contour_points)

    print(f"圆心: {center}")
    print(f"半径: {radius}")

    # 可视化点云和拟合的圆
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        circle_points.append([centroid[0], x, y])

    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(circle_points)
    circle_pcd.paint_uniform_color([1, 0, 0])

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, circle_pcd, mesh], window_name="Point Cloud and Fitted Circle")


if __name__ == "__main__":
    main()
