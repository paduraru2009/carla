import os
import numpy as np
import open3d as o3d

print("Let's draw a cubic using o3d.geometry.LineSet.")
points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
]
lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set])

exit(0)


basePath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder"
Waymo_X_orig = os.path.join(basePath, "combined_carla_moving - Copy.ply")
Waymo_X = os.path.join(basePath, "combined_carla_moving - Copy_conv.ply")

# Given a dictionary of [3d points from the cloud] -> {rgb segmentation label , raw segmentation label}, save it to the specified file
# (Taken from CARLA's code)
def save_3d_pointcloud(points_3d, filename):
    """Save this point-cloud to disk as PLY format."""

    def construct_ply_header():
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """
        points = len(points_3d)  # Total point number
        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar red',
                  'property uchar green',
                  'property uchar blue',
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        label = point[6]
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print ("Problem " + str(point))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print (point)
            print ('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

def do_conversion_toO3D(srcPath, targetPath):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(srcPath)
    nr_points = plydata.elements[0].count
    pointcloud3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
    newpointcloud3d = []
    for point in pointcloud3D:
        x,y,z,r,g,b, label = point

        """
        r /= 255.0
        g /= 255.0
        b /= 255.0
        """

        newpointcloud3d.append((x, y, z, r, g, b, label))

    save_3d_pointcloud(newpointcloud3d, targetPath)


#do_conversion_toO3D(Waymo_X_orig, Waymo_X)

def do_conv_folder(basePath, START_FRAME, END_FRAME, doSegToo = False):
    for i in range(START_FRAME, END_FRAME+1):
        print("## convering frame ", i)
        filename_in = os.path.join(basePath, "combined_carla_moving_f{0:05d}.ply".format(i))
        filename_out = os.path.join(basePath, "combined_carla_moving_f{0:05d}_conv.ply".format(i))

        print("RGB..")
        if not os.path.exists(filename_out):
            do_conversion_toO3D(filename_in, filename_out)

        print("seg..")
        if doSegToo:
            filename_in = os.path.join(basePath, "combined_carla_moving_segColor_f{0:05d}.ply".format(i))
            filename_out = os.path.join(basePath, "combined_carla_moving_segColor_f{0:05d}_conv.ply".format(i))

            if not os.path.exists(filename_out):
                do_conversion_toO3D(filename_in, filename_out)


sceneBasePath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder/Scene18311"
do_conv_folder(sceneBasePath, 0, 197, doSegToo=True)

if False:
    import open3d as o3d

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(Waymo_X)
    print(pcd)
    print(np.asarray(pcd.points))
    print(np.asarray(pcd.colors))
    print(pcd.has_colors())
    o3d.visualization.draw_geometries([pcd])