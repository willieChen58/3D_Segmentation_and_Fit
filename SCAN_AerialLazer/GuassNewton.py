import numpy as np
import open3d as o3d
from itertools import combinations


class Node:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y})"


def sqr(x):
    return x * x


def dis(x, y):
    return np.sqrt(sqr(x.x - y.x) + sqr(x.y - y.y))


def fit_circle(A, B, C):
    D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))
    if D == 0:
        return None, float('inf')  # Points are collinear
    O = Node()
    O.x = ((sqr(A.x) + sqr(A.y)) * (B.y - C.y) + (sqr(B.x) + sqr(B.y)) * (C.y - A.y) + (sqr(C.x) + sqr(C.y)) * (
            A.y - B.y)) / D
    O.y = ((sqr(A.x) + sqr(A.y)) * (C.x - B.x) + (sqr(B.x) + sqr(B.y)) * (A.x - C.x) + (sqr(C.x) + sqr(C.y)) * (
            B.x - A.x)) / D
    R = dis(O, A)
    return O, R


def calculate_loss(O, R, points):
    loss = 0
    for p in points:
        loss += sqr(dis(p, O) - R)
    return loss


def residuals(params, points):
    ox, oy, r = params
    residuals = []
    for p in points:
        residuals.append(np.sqrt(sqr(p.x - ox) + sqr(p.y - oy)) - r)
    return np.array(residuals)


def jacobian(params, points):
    ox, oy, r = params
    J = np.zeros((len(points), 3))
    for i, p in enumerate(points):
        d = np.sqrt(sqr(p.x - ox) + sqr(p.y - oy))
        if d == 0:
            J[i] = [0, 0, -1]
        else:
            J[i] = [(ox - p.x) / d, (oy - p.y) / d, -1]
    return J


def gauss_newton(params, points, max_iterations=100, tol=1e-6):
    for _ in range(max_iterations):
        r = residuals(params, points)
        J = jacobian(params, points)
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        params = params + delta
        if np.linalg.norm(delta) < tol:
            break
    return params


def main():
    # 读取点云数据
    pcd = o3d.io.read_point_cloud("pointcloud/arc_segment.ply")
    points = np.asarray(pcd.points)

    centroid = np.mean(points, axis=0)

    # 将点云数据投影到YOZ平面（使用Y和Z坐标）
    pts = [Node(point[1], point[2]) for point in points]

    # 初始化圆心和半径
    initial_params = [0, 0, 1]  # 可以根据情况进行初始化

    # 使用高斯牛顿法最小化损失函数
    optimized_params = gauss_newton(initial_params, pts)
    O = Node(optimized_params[0], optimized_params[1])
    R = optimized_params[2]

    # 输出所选的三个最优点的坐标
    print("所选的三个最优点的坐标:")
    best_loss = float('inf')
    best_comb = None
    for comb in combinations(pts, 3):
        A, B, C = comb
        O_temp, R_temp = fit_circle(A, B, C)
        if O_temp is None:  # Points are collinear
            continue
        loss = calculate_loss(O_temp, R_temp, pts)
        if loss < best_loss:
            best_loss = loss
            best_comb = comb

    for pt in best_comb:
        print(pt)

    # 输出损失函数值
    print(f"损失函数值: {best_loss:.10f}")

    print(f"Radius: {R:.10f}")
    print(f"Diameter: {R*2:.10f}")
    print(f"Center: ({O.x:.10f}, {O.y:.10f})")

    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        x = O.x + R * np.cos(angle)
        y = O.y + R * np.sin(angle)
        circle_points.append([centroid[0], x, y])

    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(np.array(circle_points))
    circle_pcd.paint_uniform_color([1, 0, 0])  # 设置点云颜色
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(10, (0, 0, 0))
    o3d.visualization.draw_geometries([pcd, mesh, circle_pcd])

    # 在Open3D中可视化点云和圆
    pcd.paint_uniform_color([0.65, 0.65, 0.65])  # 设置点云颜色
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)
    vis.add_geometry(circle_pcd)

    # 创建圆心点
    center = np.array([[centroid[0], O.x, O.y]])
    center_point = o3d.geometry.PointCloud()
    center_point.points = o3d.utility.Vector3dVector(center)
    center_point.paint_uniform_color([1, 0, 0])  # 红色

    vis.add_geometry(center_point)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
