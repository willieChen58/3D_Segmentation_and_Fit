import numpy as np
import open3d as o3d
import random


class Model:
    def __init__(self):
        self.point = np.array([0, 0, 0])
        self.direction = np.array([0, 0, 1])
        self.r = 0
        self.lIndices = []
        self.gIndices = []


def RANSAC_FIT_Cylinder(pcd, sigma, min_r, max_r, sample_num, iter):
    k = 50  # 邻近点数，计算法向量
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k))

    # 法向量重定向
    pcd.orient_normals_consistent_tangent_plane(10)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    nums = points.shape[0]

    if sample_num > nums:
        raise ValueError("采样点大于点云点数")

    range_index = list(range(nums))
    model = Model()
    pretotal = 0  # 符合拟合模型的数据个数

    for i in range(iter):
        idx = random.sample(range_index, sample_num)
        sample_data = points[idx, :]
        normal_data = normals[idx, :]

        # 拟合圆柱
        p1, p2 = sample_data[0, :], sample_data[1, :]
        n1, n2 = normal_data[0, :], normal_data[1, :]

        # 计算圆柱系数
        w = p1 - p2
        a, b, c = np.dot(n1, n1), np.dot(n1, n2), np.dot(n2, n2)
        d, e = np.dot(n1, w), np.dot(n2, w)
        denominator = a * c - b * b

        if denominator < 1e-8:
            sc, tc = (0, d / b) if b > c else (0, e / c)
        else:
            sc, tc = (b * e - c * d) / denominator, (a * e - b * d) / denominator

        line_pt = p1 + sc * n1  # 轴线上一点
        line_dir = (n1 + n2) / 2  # 轴向取两个法向量的平均
        line_dir /= np.linalg.norm(line_dir)  # 轴向归一化

        # 计算半径（点到线的距离）
        vec = p1 - line_pt
        r = np.linalg.norm(np.cross(vec, line_dir))
        if r < min_r or r > max_r:
            continue

        # 统计符合内点的数量
        dist = np.dot(points - line_pt, line_dir)
        proj_point = points - np.outer(dist, line_dir)

        # 计算距离
        dists = np.linalg.norm(proj_point - line_pt, axis=1)

        gIndices = np.where(np.abs(dists - r) <= sigma)[0]
        lIndices = np.where(dists <= r - sigma)[0] if r - sigma > 0 else []

        total = len(gIndices) - len(lIndices)

        if total > pretotal:
            pretotal = total
            model.point = line_pt
            model.direction = line_dir
            model.r = r
            model.lIndices = lIndices
            model.gIndices = gIndices

    # 将结果转换为厘米
    model.point *= 1000
    model.r *= 1000

    if model.r == 0:
        return None

    center_location = np.array([model.point[0], model.point[1], model.point[2]])

    return center_location, model.direction, model.r


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("pointcloud/cloudyuanzhu1.ply")
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, mesh])

    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)

    sigma = 0.1  # 距离拟合圆柱两侧的距离
    min_r = 0.0105  # 设置圆柱最小半径（单位：米）
    max_r = 0.0110  # 设置圆柱最大半径（单位：米）
    sample_num = 3  # 随机采样的点数
    iter = 1000  # 迭代次数

    while True:
        result = RANSAC_FIT_Cylinder(pcd, sigma, min_r, max_r, sample_num, iter)
        if result is not None:
            center_location, direction, r = result
            print(f"圆柱中心点: {center_location[0]:.2f} mm, {center_location[1]:.2f} mm, {center_location[2]:.2f} mm\n"
                  f"圆柱朝向: {direction}\n"
                  f"圆柱半径: {r:.2f} mm")

            circle_points = []
            for angle in np.linspace(0, 2 * np.pi, 100):
                x = centroid[0] + r * np.cos(angle) + r
                y = centroid[1] + r * np.sin(angle)
                circle_points.append([x, y, centroid[2]])
            print(circle_points)
            circle_pcd = o3d.geometry.PointCloud()
            circle_pcd.points = o3d.utility.Vector3dVector(np.array(circle_points))
            circle_pcd.paint_uniform_color([1, 0, 0])  # 设置点云颜色

            o3d.visualization.draw_geometries([pcd, circle_pcd])
            break
