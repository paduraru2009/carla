import numpy as np
import pickle
import os
import pytransform3d as py3d
from pytransform3d import rotations

def rotateAroundAxis(axis, degrees):
    euler = [0, 0, 0]
    euler[axis] = degrees
    R = py3d.rotations.matrix_from_euler_xyz(euler)
    return R

# TODO: these should be data driven. What if we have 1000 datasets ?
USE_WAYMO = True # False means use citiscapes

if USE_WAYMO == False:
    SCALE_FACTOR = 5.0
    SCALE_FACTOR_FOR_POSITIONS = SCALE_FACTOR
    R = rotateAroundAxis(0, np.pi)
    POINT_SIZE = 1.0
    NEEDS_AXIS_INVERSION = True
else:
    SCALE_FACTOR = 1.0
    SCALE_FACTOR_FOR_POSITIONS = 5.0 # The idea is that the environment is already scaled but positions are not
    R = np.eye(3)
    POINT_SIZE = 4.0
    NEEDS_AXIS_INVERSION = False

SHOW_COORDINATE_AXIS = True
SHOW_TRAJECTORY_WAYPOINTS = True

BACKGROUND_COLOR = [0.27, 0.27, 0.27]
INV_SCALE_FACTOR = 1.0 / SCALE_FACTOR
POSE_SCALE_FACTOR = 0.065
INV_POSE_SCALE_FACTOR = 1.0 / POSE_SCALE_FACTOR

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]

IS_DEBUG_ENABLED = True
CAMERA_PARAMS_FILE_TUBINGEN = "ScreenCamera_2020-03-12-07-54-53.json" #"ScreenCamera_2020-03-11-17-42-08.json" # "ScreenCamera_2020-03-09-15-13-20.json"


class Trajectory:
    def __init__(self, outVisFolder, trajectoryPoints, trajectorySpeeds, trajectoryColors):
        self.OUT_VIS_FOLDER = outVisFolder      # Visualization folder
        if not os.path.exists(self.OUT_VIS_FOLDER):
            os.makedirs(self.OUT_VIS_FOLDER)

        self.TRAJECTORY_POINTS = trajectoryPoints   # The trajectory
        self.TRAJECTORY_SPEEDS = trajectorySpeeds   # Speeds along trajectory
        self.TRAJECTORY_COLOR = trajectoryColors    # RGB list for this

class SceneConfig:
    def __init__(self, baseResourcesPath, cameraParamsFile, outputSaveParams):
        self.SCENE_RESOURCES_PATH   = baseResourcesPath
        self.CAMERA_PARAMS_FILE     = cameraParamsFile
        self.CAMERA_SAVE_PARAMS     = outputSaveParams



# ----------------------------------------------------------------------------

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
    x = pos[0] * SCALE_FACTOR_FOR_POSITIONS
    y = pos[1] * SCALE_FACTOR_FOR_POSITIONS
    z = pos[2]

    if NEEDS_AXIS_INVERSION:
        y = -y
        pos[0] = x
        pos[1] = z
        pos[2] = y
    else:
        pos[0] = x
        pos[1] = y
        pos[2] = z

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
# Position of the agent after simulation (from real data) frame ends
startPos_tubingen = simData_tubingen['people'][simData_tubingen['numFramesToSimulate']-1][1].squeeze()
startPos_tubingen[2] -= 3.0

# END PATH3
#######################################################################################

# DATA 3 - Waymo scene 000112. Usually the data should have the save starting point trajectory to mimic different paths, but not necessarly
#######################################################################################

# These trajectory are in Blender system coordinate open cloud format Z pointing up.
# We'll convert them to Open3D point cloud visualizer format and PFNN coordinate
# Blender space

# Add the offset paramerter
def processWaymoSimData(simdata, heightOffset):
    newSimData = {}
    agentIndexInEndFrame = None
    for frame_idx, frame_data in simdata.items():
        newSimData[frame_idx] = {}
        for entity_id, entity_data in frame_data.items():
            bbminMax = entity_data['BBMinMax']

            #x = (bbminMax[0][0] + bbminMax[0][1]) * 0.5
            #y = (bbminMax[1][0] + bbminMax[1][1]) * 0.5
            bbminMax[2][0] -= heightOffset
            bbminMax[2][1] -= heightOffset

            #if frame_idx == endSimFrame and entity_id == agentId:
            #    agentIndexInEndFrame = len(newSimData[frame_idx])
            #newSimData[frame_idx][entity_id] = bbminMax #append(np.array([x, y, z]))

    return simdata #newSimData


#sceneBasePath, cameraFileParams, trajectoryToSimulate_points, trajectoryToSimulate_speeds, saveParams, ):

def createSceneSimParams(sceneConfig : SceneConfig,
                        trajectoryConfig : Trajectory,
                        useSegmentationView : bool):

    sceneBasePath = sceneConfig.SCENE_RESOURCES_PATH
    cameraFileParams = sceneConfig.CAMERA_PARAMS_FILE
    saveParams = sceneConfig.CAMERA_SAVE_PARAMS
    trajectoryToSimulate_points = trajectoryConfig.TRAJECTORY_POINTS
    trajectoryToSimulate_speeds = trajectoryConfig.TRAJECTORY_SPEEDS

    centeringFilePath = os.path.join(sceneBasePath, "centering.p")
    centering = None
    with open(centeringFilePath, 'rb') as centeringFileHandle:
        centering = pickle.load(centeringFileHandle)
        scale = centering['scale']
        heightOffset = centering['height']

    # TODO: unique key stuff
    peoplePath = os.path.join(sceneBasePath, "people.p")
    simdata_people = pickle.load(open(peoplePath, "rb"), encoding='latin1')
    simdata_cars = pickle.load(open(os.path.join(sceneBasePath, "cars.p"), "rb"), encoding='latin1')

    # simdata_people = processWaymoSimData(simdata_people)
    # simdata_cars = processWaymoSimData(simdata_cars)

    # Position of the agent after simulation (from real data) frame ends
    agentStartFrame = END_FRAME_INDEX_ENV - 1
    #agentNameToFollow = list(simdata_people[agentStartFrame].keys())[0]  # '0U-i9Ibvlvz8tGCusPra1w'
    #minMaxPos = simdata_people[agentStartFrame][agentNameToFollow]['BBMinMax']
    #startPos = trajectoryToSimulate_points[0]  # (minMaxPos[:, 0] + minMaxPos[:, 1]) * 0.5
    #startPos[2] = minMaxPos[2][0]

    # Faked traj behind a bit
    startPos = trajectoryToSimulate_points[0].copy()
    startPos[0] -= 10.0  # Move a bit behind for start
    trajectoryToSimulate_points = np.concatenate(([startPos], trajectoryToSimulate_points))
    trajectoryToSimulate_speeds = np.concatenate(([trajectoryToSimulate_speeds[0].copy()], trajectoryToSimulate_speeds))

    saveParams["OUT_VIS_FOLDER"] = trajectoryConfig.OUT_VIS_FOLDER

    return {'people' : simdata_people, 'cars' : simdata_cars,
            'USE_SEGMENTATION_VIEW' : useSegmentationView,
            'heightOffset' : heightOffset,
            'frequency' : 17,
            'START_FRAME_INDEX_ENV' : START_FRAME_INDEX_ENV,
            'END_FRAME_INDEX_ENV' : END_FRAME_INDEX_ENV,
            'agentStartSimFrame' : AGENT_START_SIM_FRAME,
            'agentStartRenderFrame':START_FRAME_INDEX_AGENT_PFNN,
            'recordFramerate' : 17,
            'IS_FIXED_ENVIRONMENT' : False,
            'EnvironmentPointCloudPath' : sceneBasePath,
            'CameraFile' : cameraFileParams,
            "SIM_AGENT_START_POS" : startPos,
            "SIM_AGENT_TRAJECTORY" : trajectoryToSimulate_points,
            "SIM_AGENT_SPEEDS" : trajectoryToSimulate_speeds,
            "TRAJECTORY_NEEDS_TRANSFORM" : False, # Give it True if the trajectory of agent is not already scaled rotated etc
            "SAVE_PARAMS" : saveParams,
        }

# Should i view scene with segmentation colors ?
USE_SEGMENTED_VIEW = False

# Used for saving poses and displaying them at once for visualization
SAVE_POSE_HISTORY = False
MAX_POSES_IN_HIST = 150

# If you want to see any skeleton simulation activate this
IS_PFNN_ENABLED = False

# If you want to see simulation or just visualization of the environment. Used for faster views
VIEW_ENVIRONMENT_SIMULATION = True


START_FRAME_INDEX_ENV = 0 # At which frame to start the simulation of the environment
END_FRAME_INDEX_ENV = 300
START_FRAME_INDEX_AGENT_PFNN = END_FRAME_INDEX_ENV # At which frame to start the PFNN simulation
AGENT_START_SIM_FRAME = START_FRAME_INDEX_AGENT_PFNN - 5  # agentStartSimFrame is before END_SIM_FRAME SINCE IT NEEDS SOME INIT TIME TO START and look like walking not idle


# SCENES DEFINITIONS AND SIM FOR WAYMO
# ----------------------------------------------------------------------------
# SCENE 1:  Scene18311
Waymo_Scene18311 = SceneConfig(baseResourcesPath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder/Scene18311",
                               cameraParamsFile = "ScreenCamera_2020-07-08-11-21-42.json",
                               outputSaveParams = {
                                                    "ENABLED": 0,
                                                    "CROP_X": 0,
                                                    "CROP_Y": 407,
                                                    "CROP_WIDTH": 1620,
                                                    "CROP_HEIGHT": 620,
                                                    "SCALE_FACTOR": 1,
                                                    }
                               )

WAYMO18311_traj1 = Trajectory(outVisFolder="VideoWork/Output_Waymo18_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[137.25, -42.7, 31.9], [162.088, -13.4408, 31.9], [179.119, 42.1118, 31.9288], [168.11, 86.89, 31.9]]),
                              trajectorySpeeds=np.array([100, 230, 300, 300]),
                              trajectoryColors=[1.0, 1.0, 0.0])

WAYMO18311_traj2 = Trajectory(outVisFolder="VideoWork/Output_Waymo18_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[137.25, -42.7, 31.9], [181.566, -31.13, 31.9], [246.92, 50.534, 31.9], [683.585, 78.0543, 38.8533]]),
                              trajectorySpeeds=np.array([100, 230, 300, 300]),
                              trajectoryColors=[0.0, 1.0, 1.0])

WAYMO18311_traj3 = Trajectory(outVisFolder="VideoWork/Output_Waymo18_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[137.25, -42.7, 31.9], [215.143, -48.9776, 31.9], [344.87, -75.93, 31.9]]),
                              trajectorySpeeds=np.array([100, 230, 300]),
                              trajectoryColors=[0.0, 1.0, 0.0])

WAYMO18311_traj4 = Trajectory(outVisFolder="VideoWork/Output_Waymo18_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[137.25, -42.7, 31.9], [190.143, -73.36, 31.9], [209, -99.0, 31.9]]),
                              trajectorySpeeds=np.array([100, 230, 300]),
                              trajectoryColors=[0.0, 0.0, 1.0])



# SCENE 2:  Scene15646511
Waymo_Scene15646511 = SceneConfig(baseResourcesPath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder/Scene15646511",
                                   cameraParamsFile = "ScreenCamera_2020-07-08-19-20-16.json",
                                   outputSaveParams = {
                                                        "ENABLED" : 1,
                                                        "CROP_X" : 0,
                                                        "CROP_Y" : 428,
                                                        "CROP_WIDTH" : 1620,
                                                        "CROP_HEIGHT" : 595,
                                                        "SCALE_FACTOR": 1,
                                                        }
                                    )

WAYMO15646511_traj1 = Trajectory(outVisFolder="VideoWork/Output_Waymo15646511_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[55.546, 57.3221, 31.9], [209.29, 50.392, 31.9]]),
                              trajectorySpeeds=np.array([100, 230]),
                              trajectoryColors=[1.0, 1.0, 0.0])

WAYMO15646511_traj2 = Trajectory(outVisFolder="VideoWork/Output_Waymo15646511_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[55.546, 57.3221, 31.9], [140.53, 76.46, 31.9], [140.53, 177.64, 31.9]]),
                              trajectorySpeeds=np.array([100, 230, 300]),
                              trajectoryColors=[0.0, 1.0, 1.0])

WAYMO15646511_traj3 = Trajectory(outVisFolder="VideoWork/Output_Waymo15646511_traj1" + ("_seg" if USE_SEGMENTED_VIEW else ""),
                              trajectoryPoints=np.array([[55.546, 57.3221, 31.9], [69.379, 34.9, 31.9], [85.85, -58.182, 31.9], [150.53, -69.9, 31.9], [150.53, -166.9, 31.9]]),
                              trajectorySpeeds=np.array([100, 230, 300, 300, 300]),
                              trajectoryColors=[0.0, 1.0, 0.0])

#################################################################


# DEBUG ALL TRAJECTORIES FOR VISUALIZATION
# These are some trajectories to show with lines/curves all at once.
# This is used only for image visualization purposes !!!
# define them as null if you don't need them and they will not appear on the screen
SHOW_TRAJECTORIES_FOR_VIS = [WAYMO15646511_traj1.TRAJECTORY_POINTS, WAYMO15646511_traj2.TRAJECTORY_POINTS, WAYMO15646511_traj3.TRAJECTORY_POINTS]
#SHOW_TRAJECTORIES_FOR_VIS = [WAYMO18311_traj1.TRAJECTORY_POINTS, WAYMO18311_traj2.TRAJECTORY_POINTS, WAYMO18311_traj3.TRAJECTORY_POINTS, WAYMO18311_traj4_traj1.TRAJECTORY_POINTS]
SHOW_TRAJECTORIES_FOR_VIS_COLORS = [WAYMO18311_traj1.TRAJECTORY_COLOR, WAYMO18311_traj2.TRAJECTORY_COLOR, WAYMO18311_traj3.TRAJECTORY_COLOR, WAYMO18311_traj4.TRAJECTORY_COLOR]


simData_Waymo = createSceneSimParams(sceneConfig = Waymo_Scene15646511,
                                     trajectoryConfig = WAYMO15646511_traj1,
                                     useSegmentationView = USE_SEGMENTED_VIEW,
                                     )

