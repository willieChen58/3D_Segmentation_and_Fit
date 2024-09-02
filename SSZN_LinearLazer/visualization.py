import copy
import time
import numpy as np
import open3d as o3d
# import teaserpp_python
from scipy.spatial import ConvexHull

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

src_cloud_path = "./pipette_jian_cc_scale.ply"
dst_cloud_path = "./1.ply"
pcd_path = "pointcloud/1.ply"

z_threshold = 30

ture_trans_1 = np.array([[-0.519863784313, 0.251319468021, -0.816443622112, -0.028507037088],
                           [0.854007124901, 0.175649330020, -0.489713311195, 0.134735301137],
                           [0.020333277062, -0.951832890511, -0.305942356586, 0.079769857228],
                           [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])

ture_trans_2 = np.array([[0.231756508350, 0.949671208858, 0.210745245218, 0.107373140752],
                           [0.901498019695, -0.128275349736, -0.413336157799, 0.127434015274],
                           [-0.365500032902, 0.285779774189, 0.885855317116, -0.033899147063],
                           [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])


def remove_low_z_points(points, z_threshold):
    # 计算z坐标高于阈值的点的索引
    high_z_indices = np.where(points[:, 2] > z_threshold)[0]
    low_z_indices = np.where(points[:, 2] < z_threshold)[0]

    # 保留z坐标高于阈值的点
    kept_points = np.delete(points, low_z_indices, axis=0)

    return kept_points, high_z_indices

def compute_TopCenter_and_size(pcd_path, z_threshold):
    # 读取点云数据
    pcd = o3d.io.read_point_cloud(pcd_path)

    # 应用统计滤波去除离群点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    points = np.asarray(inlier_cloud.points)

    # 调用函数来过滤点云
    # 现在 filtered_points 只包含z坐标不低于阈值的点
    filtered_points, high_z_indices = remove_low_z_points(points, z_threshold)
    inlier_cloud = inlier_cloud.select_by_index(high_z_indices)  # 在已经过滤的点云中保留z坐标高于阈值的点

    # 计算点云的中心（质心）
    center = np.mean(np.asarray(inlier_cloud.points), axis=0)
    print(center[0], center[1], center[2])
    print("Point Cloud Center (Camera Coordinate System):", center)

    # 可视化去噪后的点云及其坐标系
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
    # inlier_cloud.translate(-center)
    o3d.visualization.draw_geometries([inlier_cloud, mesh_frame])

    return 0


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

def get_transformation(src_cloud_path, dst_cloud_path, ture_trans, NOISE_BOUND, N_OUTLIERS, OUTLIER_TRANSLATION_LB, OUTLIER_TRANSLATION_UB):
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    # Load bunny ply file
    # src_cloud = o3d.io.read_point_cloud("../example_data/bun_zipper_res3.ply")
    src_cloud = o3d.io.read_point_cloud(src_cloud_path)
    src = np.transpose(np.asarray(src_cloud.points))
    N = src.shape[1]

    dst_cloud = o3d.io.read_point_cloud(dst_cloud_path)
    dst = np.transpose(np.asarray(dst_cloud.points))

    # Add some noise
    dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND

    # Add some outliers
    outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
    for i in range(outlier_indices.size):
        shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
        dst[:, outlier_indices[i]] += shift.squeeze()

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()
    
    pred_trans = np.eye(4)
    pred_trans[:3, :3] = solution.rotation
    pred_trans[:3, 3] = solution.translation
    print(pred_trans) 
    

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Expected rotation: ")
    print(ture_trans[:3, :3])
    
    print("Estimated rotation: ")
    print(solution.rotation)
    print("Error (rad): ")
    print(get_angular_error(ture_trans[:3,:3], solution.rotation))

    print("Expected translation: ")
    print(ture_trans[:3, 3])
    
    print("Estimated translation: ")
    print(solution.translation)
    print("Error (m): ")
    print(np.linalg.norm(ture_trans[:3, 3] - solution.translation))

    print("Number of correspondences: ", N)
    print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)
    
    return pred_trans
    

def get_visualization(src_cloud_path, dst_cloud_path, esitmated_T):
    src_cloud = o3d.io.read_point_cloud(src_cloud_path)
    src_cloud_copy = copy.deepcopy(src_cloud)
    
    dst_cloud = o3d.io.read_point_cloud(dst_cloud_path)
    
    src_cloud.transform(esitmated_T)
    o3d.visualization.draw_geometries([src_cloud_copy, src_cloud, dst_cloud])


if __name__ == "__main__":

    compute_TopCenter_and_size(pcd_path, z_threshold)
    
    # estimation_T = get_transformation(src_cloud_path, dst_cloud_path, ture_trans_1, NOISE_BOUND, N_OUTLIERS, OUTLIER_TRANSLATION_LB, OUTLIER_TRANSLATION_UB)
    #
    # get_visualization(src_cloud_path, dst_cloud_path, estimation_T)

    



 

