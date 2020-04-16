import numpy as np
import pickle

SHOW_COORDINATE_AXIS = False
SHOW_TRAJECTORY_POINTS = False
POINT_SIZE = 1.0
BACKGROUND_COLOR = [0.27, 0.27, 0.27]
SCALE_FACTOR = 5.0
INV_SCALE_FACTOR = 1.0 / SCALE_FACTOR
POSE_SCALE_FACTOR = 0.065
INV_POSE_SCALE_FACTOR = 1.0 / POSE_SCALE_FACTOR
parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]

OUT_VIS_FOLDER="VideoWork/Output"
IS_DEBUG_ENABLED = "False"
CAMERA_PARAMS_FILE = "ScreenCamera_2020-03-12-07-54-53.json" #"ScreenCamera_2020-03-11-17-42-08.json" # "ScreenCamera_2020-03-09-15-13-20.json"

START_FRAME_INDEX = 0 # At which frame to start the simulation of the environment
START_FRAME_INDEX_AGENT = 30 # At which frame to start the PFNN simulation

SkeletonColors = np.array([[255,     0,    85],
[170,     0,   255],
[255,     0,   170],
[ 85,     0,   255],
[255,     0,   255],
[170,   255,     0],
[255,    85,     0],
[ 85,   255,     0],
[255,   170,     0],
[  0,   255,     0],
[255,   255,     0],
[  0,   170,   255],
[  0,   255,    85],
[  0,    85,   255],
[  0,   255,   170],
[  0,     0,   255],
[  0,   255,   255],
[255,     0,     0],
[255,     40,     40],
[  0,     0,   255],
[  0,     0,   255],
[  0,     0,   255],
[  0,   255,   255],
[  0,   255,   255],
[  0,   255,   255],
[0, 85, 170],
[0, 0, 255],
[0, 170, 255],
[0, 225, 255],
[0, 170, 85],
[0, 85, 255]], dtype=float)

SkeletonColors /= 255.0

# Color Bounding boxes for cars and pedestrians
Cars_Colors = np.array([[255,     0,    0],
[170,     0,   255],
[255,     0,   170],
[ 85,     0,   255],
[255,     0,   255],
[170,   255,     0],
[255,    85,     0],
[ 85,   255,     0],
[255,   170,     0],
[  0,   255,     0],
[255,   255,     0],
[  0,   170,   255],
[  0,   255,    85],
[  0,    85,   255]], dtype=float)

Cars_Colors /= 255.0

Pedestrians_Colors = np.array([[0,     0,     255],
[  255,   255,   0],
[  0,     0,   255],
[  0,     0,   255],
[  0,   255,   255],
[  0,   255,   255],
[  0,   255,   255],
[0, 85, 170],
[0, 0, 255],
[0, 170, 255],
[0, 225, 255],
[0, 170, 85]], dtype=float)

Pedestrians_Colors /= 255.0

# Convert from tensorflow output model to blender (or colmap) space system
def convertFromAgentModelToBlenderSpace(p):
    original_p = p.copy()
    scale = 0.709636051501
    middle = [-0.54912938, 1.82588005, -5.86619122]
    P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    pos_y = -128 / 2

    p[1] = p[1] + pos_y
    p = np.reshape(p[[2, 1, 0]], (3, 1)) * (0.2 / scale)
    p_n = np.matmul(P, p)
    p_n = p_n + np.reshape(middle, (3, 1))
    return p_n

def DEBUG_LOG(str):
    if IS_DEBUG_ENABLED:
        print(str)

############ CONVERSION FUNCTION NEEDEDED BECAUSE OF THE 3 coordinate systems: PFNN, BLENDER and CloudVis
# PFNN and CloudVis have Z axis forward, Y being UP while blender has Z as UP axis, Y being forward
# CloudVis vs Blender: is rotated around X axis with  180 deg. Check line: R = rotateAroundAxis(0, np.pi). And Scaled with SCALE_FACTOR
# THE CLOUD POINT DATA IS ROTATED AROUND Y AXIS and SCALED DIFFERENTLY
def convert2DPosFromPFnnToPointCloudVisualizer(pfnnX, pfnnZ):
    spaceX = pfnnX * POSE_SCALE_FACTOR
    spaceY = pfnnZ * POSE_SCALE_FACTOR #
    return spaceX, spaceY

def convert2DPosFromPointCloudVisualizerToPFnn(xpos, zpos):
    xpos *= INV_POSE_SCALE_FACTOR
    zpos *= INV_POSE_SCALE_FACTOR
    return xpos, zpos

def convertBlenderToPointCloudVisualizer(pos):
    x = pos[0] * SCALE_FACTOR
    y = -pos[1] * SCALE_FACTOR
    z = pos[2]
    pos[0] = x
    pos[1] = z
    pos[2] = y

# Convert the trajectory points from raw to scene coordinates (e.g. scale them, invert axis, etc)
def transformTrajectoryPointsFromBlenderToPointCloudVis(trajectoryPoints):
    for pos in trajectoryPoints:
        convertBlenderToPointCloudVisualizer(pos)

#################################### DATA TRAJECTORIES ##################################

# DATA 1 - Aachen 000042. Usually the data should have the save starting point trajectory to mimic different paths, but not necessarly
#######################################################################################
pointCloudDataPath_aachen000042 = "VideoWork/aachen_000042_copy/meshed.ply"
# These trajectory are in Blender system coordinate open cloud format Z pointing up.
# We'll convert them to Open3D point cloud visualizer format and PFNN coordinate
# Blender space

# BEGIN PATH1
trajectory_aachen000042_1 = np.array([
                       [-0.17, 1.3, -0.55],
                       [-1.0, 2.2245, -0.55],
                       [-2.6425, 2.824, -0.55],
                       [-3.2722, 4.23139, -0.55],
                       [-4.3109, 5.9, -0.55],
                       [-5.1831, 6.08889, -0.55],
                       [-6.51539, 6.088895, -0.55],])

# Should have the same size as above
speeds_aachen000042_1 = np.array([300, 300, 200, 150, 130, 130, 130]) # cm/second
startPos_aachen000042_1 = convertBlenderToPointCloudVisualizer(np.array([-0.17, -8.0, -0.55])) # Blender space
assert(len(speeds_aachen000042_1) == len(trajectory_aachen000042_1))
# END PATH1

# BEGIN PATH2
trajectory_aachen000042_2 = np.array([
                       [-0.17, 1.3, -0.480711],
                       [0.63, 7.35, 0.0],
                       [1.9268, 12.26, 0.5],
                       [2.183, 17.274, 0.5]])

# Should have the same size as above
speeds_aachen000042_2 = np.array([350, 320, 200, 130]) # cm/second
startPos_aachen000042_2 = convertBlenderToPointCloudVisualizer(np.array([-0.17, -8.0, -0.55])) # Blender space
assert(len(speeds_aachen000042_2) == len(trajectory_aachen000042_2))
# END PATH2

# BEGIN PATH3
trajectory_aachen000042_3 = np.array([
                       [-0.17, 1.3, -0.480711],
                       [-5.22435, 6.56143, 0.492362],
                       [-5.36563, 13.28604, 0.5],
                       [-4.749, 13.9, 0.5],
                       [-2.81, 20, 0.5]])

# Should have the same size as above
speeds_aachen000042_3 = np.array([350, 200, 350, 160, 140]) # cm/second
startPos_aachen000042_3 = convertBlenderToPointCloudVisualizer(np.array([-0.17, -8.0, -0.55])) # Blender space
assert(len(speeds_aachen000042_3) == len(trajectory_aachen000042_3))

DATA_SIM_PATH_aachen000042 = "VideoWork/aachen_000042_copy/agent_aachen_0000420_0.p"
# END PATH3
#######################################################################################



# DATA 2 - Tubingen 000112. Usually the data should have the save starting point trajectory to mimic different paths, but not necessarly
#######################################################################################
pointCloudDataPath_Tubingen_000112 = "VideoWork/tubingen_000112/meshed.ply"
# These trajectory are in Blender system coordinate open cloud format Z pointing up.
# We'll convert them to Open3D point cloud visualizer format and PFNN coordinate
# Blender space

# BEGIN PATH1
trajectory_tubingen_000112_1 = np.array([[-6.43, 9.8, -1.3],
                                         ])

# Should have the same size as above
speeds_tubingen_000112_1 = np.array([230]) # cm/second
assert(len(speeds_tubingen_000112_1) == len(trajectory_tubingen_000112_1))
# END PATH1

# BEGIN PATH2
trajectory_tubingen_000112_2 = np.array([
                       [-7.94, 11.753, -0.7803],
                        ])

# Should have the same size as above
speeds_tubingen_000112_2 = np.array([150]) # cm/second
assert(len(speeds_tubingen_000112_2) == len(trajectory_tubingen_000112_2))
# END PATH2


# BEGIN PATH3
trajectory_tubingen_000112_3 = np.array([
                        [3.06, 19.164, -0.7803],
                        [12.643, 5.168, -1.399],
                        ])
speeds_tubingen_000112_3 = np.array([130, 270]) # cm/second
assert(len(speeds_tubingen_000112_3) == len(trajectory_tubingen_000112_3))
# END PATH3

# BEGIN PATH4
trajectory_tubingen_000112_4 = np.array([
                        [3.06, 19.164, -0.7803],
                        [13.059, 19.588, -1]
                        ])
speeds_tubingen_000112_4 = np.array([130, 150]) # cm/second
assert(len(speeds_tubingen_000112_4) == len(trajectory_tubingen_000112_4))
# END PATH4

#DATA_SIM_PATH_aachen000042 = "VideoWork/aachen_000042_copy/agent_aachen_0000420_0.p"

# Create sim dictionary for TUBINGEN 000112 SCENE
def createSceneForTubingen():

    peoplePath = "VideoWork/tubingen_000112/people_tubingen_112.p" # "VideoWork/tubingen_000112/agent_tubingen_0001120_0.p"
    simdata_people = pickle.load(open(peoplePath, "rb"), encoding='latin1')
    for frame in simdata_people:
        for entityPos in frame:
            entityPos = entityPos.squeeze()
            entityPos += np.array([0.0, 0.0, -5.0]) # A little translation :)
    simdata_cars = pickle.load(open("VideoWork/tubingen_000112/cars_tubingen_112.p", "rb"), encoding='latin1')

    '''
    for f in range(30):
        for i in range(len(people_list[f])):

            bbox_agentSpace = people_list[f][i]
            #bbox_agentSpace[1] -= 12

            minPos_agentSpace = bbox_agentSpace[:, 0]
            maxPos_agentSpace = bbox_agentSpace[:, 1]
            minPos_blenderSpace = convertFromAgentModelToBlenderSpace(minPos_agentSpace)
            maxPos_blenderSpace = convertFromAgentModelToBlenderSpace(maxPos_agentSpace)

            people_list[f][i][:,0] = minPos_blenderSpace.squeeze()
            people_list[f][i][:,1] = maxPos_blenderSpace.squeeze()
    #print(people_list)
    '''

    return {'people' : simdata_people, 'cars' : simdata_cars, 'frequency' : 17, 'numFramesToSimulate' : 30,
            'agentStartSimFrame' : 25, 'agentStartRenderFrame':30, 'recordFramerate':17}

simData_tubingen = createSceneForTubingen()
startPos_tubingen = simData_tubingen['people'][simData_tubingen['numFramesToSimulate']-1][1].squeeze()
startPos_tubingen[2] -= 3.0

# END PATH3
#######################################################################################

