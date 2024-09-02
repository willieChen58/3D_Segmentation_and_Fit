import open3d as o3d
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

def load_point_cloud(file_path):
    # 加载点云数据
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def main():
    file_path = 'pointcloud/22.ply'  # 替换为你的点云文件路径
    pcd = load_point_cloud(file_path)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, mesh], window_name="Point Cloud and Fitted Circle")


if __name__ == "__main__":
    main()
