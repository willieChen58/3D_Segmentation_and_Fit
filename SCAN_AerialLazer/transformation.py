import trimesh
import open3d as o3d


aix = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

pcd = o3d.io.read_point_cloud("pointcloud/bottle_body.ply")

o3d.visualization.draw_geometries([pcd, aix], "原位置")

mesh = trimesh.load("pointcloud/bottle_body.ply")
Tform = mesh.apply_obb()
mesh.export("pointcloud/obb_bottle_body.ply", encoding="ascii")

pcd = o3d.io.read_point_cloud("pointcloud/obb_bottle_body.ply")
o3d.visualization.draw_geometries([pcd, aix], "转换后")