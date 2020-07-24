import os
import numpy as np
import open3d as o3d
from pytransform3d import rotations
import copy
import pytransform3d as py3d

def rotateAroundAxis(axis, degrees):
    euler = [0, 0, 0]
    euler[axis] = degrees
    R = py3d.rotations.matrix_from_euler_xyz(euler)
    return R

basePath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder"
#Waymo_X_orig = os.path.join(basePath, "combined_carla_moving - Copy.ply")
Waymo_X = os.path.join(basePath, "combined_carla_moving - Copy_conv_2.ply")#"combined_carla_moving - Copy_conv.ply")


if True:
    import open3d as o3d


    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(Waymo_X)
    print(pcd)
    print(np.asarray(pcd.points))
    print(np.asarray(pcd.colors))
    print(pcd.has_colors())

    R = rotateAroundAxis(0, np.pi)
    pcd = pcd.rotate(R, center=np.array([0, 0, 0]))
    #pcd = pcd.scale(SCALE_FACTOR, center=np.array([0, 0, 0]))

    print("Testing IO for meshes ...")

    objmesh_path = os.path.join(basePath, "sample_mesh.obj")
    skinned_mesh = o3d.io.read_triangle_mesh(objmesh_path)
    skinned_mesh.compute_vertex_normals()
    skinned_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    skinned_mesh.scale(scale=4.0, center = skinned_mesh.get_center())


    Rx_mesh = rotateAroundAxis(0, np.pi/2)
    Ry_mesh = rotateAroundAxis(2, np.pi/2)
    skinned_mesh = skinned_mesh.rotate(Rx_mesh, center=skinned_mesh.get_center())
    skinned_mesh = skinned_mesh.rotate(Ry_mesh, center=skinned_mesh.get_center())
    skinned_mesh.translate([15, 0, -11])

    mesh_vertices = np.asarray(skinned_mesh.vertices)
    mesh_triangles = np.asarray(skinned_mesh.triangles)
    mesh_colors = np.asarray(skinned_mesh.vertex_colors)

    #print(mesh)

    skinnedmesh_sampled = skinned_mesh.sample_points_uniformly(number_of_points=800)
    skinnedmesh_sampled.translate([15, 0, 0])
    skinnedmesh_sampled.paint_uniform_color([0.0, 0.0, 0.1])
    skinnedmesh_sampled_bbox = skinnedmesh_sampled.get_axis_aligned_bounding_box()
    o3d.visualization.draw_geometries([pcd, skinned_mesh, skinnedmesh_sampled, skinnedmesh_sampled_bbox])


    #o3d.visualization.draw_geometries([mesh_cylinder, mesh_frame, mesh_box, mesh_sphere
    #                                   , mesh
    #                                   ])