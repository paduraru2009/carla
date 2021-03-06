import numpy as np
import os.path, math
import os
import copy

#from utils.constants import Constants
import glob
import pickle
import subprocess

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

# Carla Client initialization and help functions
try:
    sys.path.insert(0,glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


'''
control = carla.WalkerControl()
control.speed = 10
import numpy as np
R = 1 + np.arange(10)
Rlist = list(R)
print(type(Rlist))
print(type(R.tolist()))
control.poses = R.tolist() # [1.0, 2.0, 3.0]
#control.poses[0] = 0
#control.poses[1] = 2
'''

import logging
import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import queue as queue


def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

############################################################################ Values to set


timestamp= "2020-03-01-21-57-41.449517" #"2020-03-02-14-56-59.676875" #  "2020-03-01-21-57-41.449517" # Timestamp of run to visualize
path="files_trial/" # Path to settings files from the drive
env_path="files_trial/scenes/" # Path to the environment files from the drive
# Path to statistics files
stat_path="files_trial/" #"/home/mariap/Results/statistics/val/"
out_path="out" # Path to save output to.

PEDESTRIAN_OFFSET_Z = 1.3 # Diff on Z axis Between our data and Carla map

TARGET_NBR = 4 # The starting position ID in Carla space (use convert NBR to convert from data reading to Carla env - multiplying by 4 because they were too close)
TARGET_EPOCH_ID = 9
FRAME_RECORDING_STOP_FRAME = 225 # In which frame to stop recording

actor_perspective=True
DONT_CREATE_OTHER_AGENTS = False # If activated, this will spawn only the main test agent
LOAD_ONLY_NEEDED_DATA = True
USE_PFNN_VISUALIZER = True
PEDESTRIAN_BLUEPRINT_FOR_MAIN_ACTOR = 'walker.pedestrian.0004'if USE_PFNN_VISUALIZER == True else 'walker.pedestrian.0003'

HACK_TO_EXPORT_POSES = True

def convertFileNbrToCarlaNbr(nbr):
    return (int(nbr) * 4)


# Main starts further down.

################################################################################ Definition of help functions.

epoch=3000# update_frequency*100

settings_ending="settings.txt"

# Reads settings of run.
def read_settings_file(path):

    # Find settings file
    settings_file = glob.glob(path + timestamp + "*" + settings_ending)
    print(("Setting file "+str(settings_file)))
    if len(settings_file) == 0:
        settings_file = glob.glob(path + timestamp + "*" + settings_ending)
    if len(settings_file)==0:
        return [],[],[],[],[],[]

    # Get movie name
    name_movie = os.path.basename(settings_file[0])[len(timestamp) + 1:-len("_settings.txt")]
    target_dir = path + name_movie
    #subprocess.call("cp " + settings_file[0] + " " + target_dir + "/", shell=True)
    semantic_channels_separated = False
    in_2D=False
    num_measures = 6
    with open(settings_file[0]) as s:
        for line in s:
            if "Semantic channels separated" in line:
                if line[len("Semantic channels separated: "):].strip()=="True":
                    semantic_channels_separated = True

            if "Minimal semantic channels : " in line:
                if line[len("Minimal semantic channels : "):].strip() == "True":
                    mini_labels = True
            if "Number of measures" in line:
                num_measures = int(line[len("Number of measures: "):])
            if "2D input to network:"in line:
                if line[len("2D input to network: "):].strip()== "True":
                    in_2D=True
                print((line[len("2D input to network: "):].strip() +" "+str(bool(line[len("2D input to network: "):].strip()))+" "+str(in_2D)))

    return {}, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D



init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

# Read camera matrix from file
def get_camera_matrix(filepath):
    cameras_path = os.path.join(filepath, 'cameras.p')
    cameras_dict = pickle.load(open(cameras_path, "rb"), encoding='latin1')
    frames=sorted(cameras_dict.keys())
    print (cameras_path)

    frame = frames[-1]#np.min(cameras_dict.keys())
    R_inv = cameras_dict[frame]['inverse_rotation'] # Transforms world to camera
    middle_n = R_inv[0:3, 3]
    R = np.transpose(R_inv[0:3, 0:3])
    middle_3 = -np.matmul(R, middle_n)
    #return R, middle, frames
    t=R_inv[0:3, 3]
    C= cameras_dict[frame]['camera_to_car']
    C2=np.matmul(C,R_inv )
    middle_w=-np.matmul(C2[0:3,0:3], C2[0:3,3])
    return R_inv[0:3, 0:3], t, frames, middle_3 ,C, middle_w # , car_pos


# Combine statistics files.
def get_scenes(test_files,test_points):

    scenes=[]

    stats_temp = []
    prev_test_pos = 0

    for j, pair in enumerate(sorted(test_files)):
        # Load only the data we are interested in from the testing scenario
        thisFileToCarlaNbr = convertFileNbrToCarlaNbr(j)
        if LOAD_ONLY_NEEDED_DATA and thisFileToCarlaNbr != TARGET_NBR:
            continue

        if prev_test_pos != test_points[pair[0]]:
            #print 'switch'
            for init_m, stats in enumerate(stats_temp):
                if len(stats)>0:
                    scenes=stats_temp
            stats_temp = []

        cur_stat = np.load(pair[1])
        poses_path=pair[1][:-len(".npy")]+"_poses.npy"
        #print poses_path
        pos_stat=np.load(poses_path)

        for ep_nbr in range(cur_stat.shape[0]):

            if LOAD_ONLY_NEEDED_DATA and ep_nbr != TARGET_EPOCH_ID:
                continue

            #if int(cur_stat[ep_nbr, 0, 38 + 11])!=2:
            # agent_pos = cur_stat[ep_nbr, :, 0:3]
            # agent_probabilities = cur_stat[ep_nbr, :, 7:34]
            # agent_reward = cur_stat[ep_nbr, :, 34]
            # agent_measures = cur_stat[ep_nbr, :, 38:]
            # # car hit
            # out_of_axis = cur_stat[ep_nbr, -1, 0]
            # if out_of_axis==0:
            #print "File "+str(pair[1])+" "+str(ep_nbr)
            stats_temp.append([cur_stat[ep_nbr, :, :], pair[1], ep_nbr, pos_stat[ep_nbr,:,:]])
        prev_test_pos = test_points[pair[0]]

    scenes = stats_temp
    return stats_temp

# Get car's rotation matrix
def get_car_rotation_matrix(yaw):
    cy = math.cos(np.radians(yaw))
    sy = math.sin(np.radians(yaw))
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    matrix=np.matrix(np.identity(3))
    matrix[0, 0] =cp * cy
    matrix[0, 1] =cy * sp * sr - sy * cr
    matrix[0, 2] = - (cy * sp * cr + sy * sr)
    matrix[1, 0] =  sy * cp
    matrix[1, 1] =sy * sp * sr + cy * cr
    matrix[1, 2] =cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -(cp * sr)
    matrix[2, 2] = cp * cr

    return matrix
P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # camera to vehicle

# Find bounding box of car
def bbox_of_car(K, R,  car_id, find_poses, frame_cars, height, image_size, middle, scale):
    pos_car = frame_cars[car_id]['transform']

    bbox_car = np.squeeze(np.asarray(np.matmul(np.matmul(R, get_car_rotation_matrix(frame_cars[car_id]['yaw'])),
                                               frame_cars[car_id]['bounding_box'])))
    point_0 = np.squeeze(np.asarray(np.matmul(R, np.array(pos_car).reshape((3, 1))) + middle))
    boxes_2d=[]
    if find_poses:
        x_values = []
        y_values = []
        bbx = [np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]])]
        for point in bbx:
            point_2D = np.matmul(K, point.reshape((3, 1)))
            point_2D = point_2D / point_2D[2]
            x_2d = int(image_size[0] - point_2D[0])
            y_2d = int(image_size[1] - point_2D[1])
            x_values.append(x_2d)
            y_values.append(y_2d)
        boxes_2d.append([y_values, x_values])
    point_0 = np.squeeze(np.asarray(np.matmul(R, np.array(pos_car).reshape((3, 1))) + middle))
    bbox_car = bbox_car * scale
    point_0 = point_0 * scale
    bbox_car = bbox_car * [-1, -1, 1]  # normal camera coodrinates
    point_0 = point_0 * [-1, -1, 1]  # normal camera coodrinates
    point_0 = np.squeeze(np.asarray(np.matmul(P_inv, point_0)))
    point_0[2] = point_0[2] - height
    bbox_car = np.squeeze(np.asarray(np.matmul(P_inv, bbox_car)))
    car = np.column_stack((point_0 - np.abs(bbox_car), point_0 + np.abs(bbox_car)))
    return car, boxes_2d


def sort_files_eval(files):

    test_files = []
    reconstructions_test=[]
    poses_test = {}
    nbr_files = 0

    special_cases=[]
    numbers_people = []

    for agent_file in sorted(files):

        basename = os.path.basename(agent_file)


        nbrs = basename.strip()[:-len('.npy')]

        vals = nbrs.split('_')

        #print vals
        try:
            pos=int(vals[-1])
        except ValueError:
            try:
                pos = int(vals[-1][:-len("reconstruction")])
            except ValueError:
                # try:
                #     pos = int(vals[-2])
                #     poses_test.append((int(vals[-4]), agent_file, pos)
                # except ValueError:
                print (basename)
                pos = -1
                pos=-1

        if pos>=0:

            if not "reconstruction" in vals[-1]:
                test_files.append((int(vals[-3]), agent_file, int(vals[-1])))
            else:

                reconstructions_test.append((int(vals[-3]), agent_file, int(vals[-1][:-len("reconstruction")])))

            nbr_files += 1
    return test_files, reconstructions_test, poses_test


def agentPosFromModelToCARLA(agentPos, pos_y, centering, R, P, middle, frame):
    p = agentPos.copy()
    if frame == 0:
        print(("point " + str(p)))
    p[1] = p[1] + pos_y
    if frame == 0:
        print(("point+y " + str(p)))
    p[0] = p[0] + centering["height"]
    if frame == 0:
        print(("point+ height " + str(p)))

    p = np.reshape(p[[2, 1, 0]], (3, 1)) * (1.0 / centering['scale'])
    if frame == 0:
        print(("point scaled " + str(p)))
    p = np.matmul(P, p)
    if frame == 0:
        print(("point P" + str(p)))
    p = p * np.reshape([-1, -1, 1], (3, 1))
    if frame == 0:
        print(("point p " + str(p)))
    p = p - np.reshape(middle, (3, 1))
    if frame == 0:
        print(("point middle " + str(p)))
    p = np.matmul(np.transpose(R), p)
    if frame == 0:
        print(("point R " + str(p)))

    return p

def agentPosFromPFNNToModel(possInPfnn, RotationFromPFNNTomodel, originPos):
    y = possInPfnn[2]
    x = possInPfnn[0]
    current_pos = np.array([y, x]).transpose()

    pos_exact=copy.copy(originPos)
    target_rotated = np.matmul(RotationFromPFNNTomodel, current_pos)
    pos_exact[1] = target_rotated[0] / 20.0 + originPos[1]
    pos_exact[2] = target_rotated[1] / 20.0 + originPos[2]
    pfnnPosInModelSpace = pos_exact #np.round(pos_exact) #.astype(int) # PFNN position in my coordinate system
    return pfnnPosInModelSpace

################################################################### Main starts here!!!!!

P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

init_names= {-1: "training", 0: "training", 1: "On pedestrian", 3: "On pavement", 2: "By car", 4: "Random",5: "car_env", 6: "Near pedestrian",7:"pedestrian environment", 9:"average"}

pos_y=-128 / 2

labels_to_use, name_movie, target_dir, num_measures, semantic_channels_separated, in_2D=read_settings_file(path)

# Find CARLA dataset
filename_list = {}
# Get files to run on.
ending_local = "test_*"
for filepath in glob.glob(env_path + ending_local):
    parts = os.path.basename(filepath).split('_')
    pos = int(parts[-1])
    filename_list[pos] = filepath


# find statistics files.
find_path=stat_path+'*'+timestamp+"*"

match=find_path+"*.npy"

files=glob.glob(match)

test_files, reconstructions_test, poses_test = sort_files_eval(files)
#filenames_itr_cars, test_files_cars, filenames_itr_people,test_files_people,  filenames_itr, test_files, reconstructions_test, iterations, iterations_cars, special_cases
#
# test_files=[(0, os.path.join(path, 'visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_0.npy'), 0),
#  (0, os.path.join(path,'visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_1.npy'), 1),
#  (0, os.path.join(path,'visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_2.npy'), 2),
#  (0, os.path.join(path,'visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_3.npy'), 3)]
#test_files=[ (0, os.path.join(path,'visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_2.npy'), 2)]

print(("Test  files: "+str(test_files)))
test_points = {0: 0}


epoch = 0
saved_files_counter = 0
csv_rows = []

if len(test_files)>0:
    scenes =get_scenes(test_files,test_points)

for scene in scenes:
    filepath_npy = scene[1]
    #filepath_npy = os.path.join(path, "visualize_2D_goal_agent_2019-11-13-22-28-15.914025_test_0_-64_2.npy")
    epoch = scene[2]
    cur_stat = scene[0]


    #file_nbrs = [0, 8, 24, 36] # Which Carla initialization point to look at. Depends on the above file.
    #if test_data:
    nbr = filepath_npy.split('_')[-1]
    nbr = nbr[:-len('.npy')]
    nbr = convertFileNbrToCarlaNbr(nbr)
    # epoch = 13
    # cur_stat = scene[0]
    agent_pos = cur_stat[:, 0:3] # Agent's trajectory
    agent_vel = cur_stat[:, 3:6]  # Actions taken by agent
    agent_action = cur_stat[ :, 6] # Actions taken by agent
    agent_probabilities = cur_stat[:, 7:34] # Probabilities of different actions (output of RL agent)
    agent_reward = cur_stat[:, 34] # Agent's reward
    agent_measures = cur_stat[:, 38:] # Different measures of agent
    agent_goals = cur_stat[3:5, -1] # Goal position of agent
    reached_goal = agent_measures[:,13] # Goal reached by agent
    yaw_mapping=[-0.75, -0.5,-0.25,-1,0,0.25,0.75,0.5,0] # yaw for different actions.

    poses_stats=scene[3]
    agent_poses=poses_stats[ :, :93]
    agent_pfnn_pos=poses_stats[ :, 93:95]
    agent_pfnn_itr=poses_stats[ :, 95]
    agent_pfnn_avg_speed=poses_stats[ :, 96]

    # print filepath_npy
    print (str(nbr) +" "+str(epoch))
    print (filename_list[int(nbr)] + " " + str(nbr))
    if int(nbr)==TARGET_NBR and int(epoch) == TARGET_EPOCH_ID:
        filepath = filename_list[int(nbr)]
        basename = os.path.basename(filepath)
        print(("Filepath: "+str(filepath_npy)))
        print (epoch)

        # Load car and pedestrian positions
        cars_path = os.path.join(filepath, 'cars.p')
        cars_dict = pickle.load(open(cars_path, "rb"), encoding='latin1')
        print (cars_path)

        people_path = os.path.join(filepath, 'people.p')
        people_dict = pickle.load(open(people_path, "rb"), encoding='latin1')
        print (people_path)

        # Load how my coordinate system is related to CARLA coordinate system.
        centeringPath = os.path.join(filepath, "centering.p")
        centering = pickle.load(open(centeringPath, "rb"), encoding='latin1')
        print(centeringPath)

        startPositionsPath = os.path.join(path, "start_positions.p")
        start_poses = pickle.load(open(startPositionsPath, "rb"), encoding='latin1')
        print (startPositionsPath)

        R, middle, frames, middle_2,C, middle_w = get_camera_matrix(filepath)


        #grid_axis_x=[]
        # for y in range(128):
        #     p=[]
        ####################################################################
        # Get pose coordinate system rotation for PFNN network. The first action is where the z axis is rotated.


        rotation_matrix = np.zeros((2, 2), np.float)
        inverse_rotation_matrix = np.zeros((2, 2), np.float)

        init_pos=agent_pos[0,:].copy()
        y = (agent_vel[0,1])
        z = (agent_vel[0,1])
        d = np.sqrt(y ** 2 + z ** 2)
        if d > 0:
            # Rotation matrix from my position to PFNN
            rotation_matrix[0, 0] = y / d
            rotation_matrix[0, 1] = z / d
            rotation_matrix[1, 1] = y / d
            rotation_matrix[1, 0] = -z / d
            # Rotation matrix from PFNN to my coordinate system


            '''
            inverse_rotation_matrix[0, 0] = y / d
            inverse_rotation_matrix[0, 1] = -z / d
            inverse_rotation_matrix[1, 1] = y / d
            inverse_rotation_matrix[1, 0] = z / d
            '''

            '''
            inverse_rotation_matrix[0, 0] = 1.0
            inverse_rotation_matrix[0, 1] = 0.0
            inverse_rotation_matrix[1, 1] = 1.0
            inverse_rotation_matrix[1, 0] = 0.0
            '''


            inverse_rotation_matrix = rotation_matrix


        print ("Velocity "+str(agent_vel[0,:]))
        print (rotation_matrix)

        print ("Agent init positions: "+str(init_pos)+" "+str(agent_pos[1,:]))
        ################################################################
        agent_positions=[]
        agent_pose_perFrame=[]
        agent_pose_perFrame_inPFNNLocal = []
        agent_yaw=[]
        previous_pos=[0,0]

        agent_goal = agentPosFromModelToCARLA(np.array([0, agent_goals[0], agent_goals[1]]), pos_y, centering, R, P, middle, 0) # The goal point in CARLA coordinate system
        directions=[]
        directions_angle=[]

        for frame in range(agent_pos.shape[0]):
            if reached_goal[max(frame-15,0)]==0:
                agentPosThisFrame_modelSpace = agent_pos[frame, :]

                ####################################
                # Convert PFNN position into CARLA position
                pfnn_itr = int(agent_pfnn_itr[frame])
                #print pfnn_itr

                # Get agent pose in PFNN coordinate system
                pfnn_pos = agent_pfnn_pos[pfnn_itr, :].copy()
                pfnn_pose = agent_poses[pfnn_itr, :].copy()

                # HACK TO SAVE POSE - NOT NEEDED
                pfnn_pose_copy = pfnn_pose.copy()
                pfnn_offset_X = pfnn_pos[0]
                pfnn_offset_Z = pfnn_pos[1]
                for i in range(31):
                    pfnn_pose_copy[i*3 + 0] -= pfnn_offset_X
                    pfnn_pose_copy[i*3 + 2] -= pfnn_offset_Z

                agent_pose_perFrame_inPFNNLocal.append(pfnn_pose_copy)
                #END HACK

                # Convert PFNN pos to model space
                pfnnPosInModelSpace = agentPosFromPFNNToModel(np.array([pfnn_pos[0], 0.0, pfnn_pos[1]]), inverse_rotation_matrix, init_pos)

                # Convert PFNN pos from model space to CARLA
                pfnnPosInCARLASpace = agentPosFromModelToCARLA(pfnnPosInModelSpace, pos_y, centering, R, P, middle, frame)

                # Because there is a gap on Z axis between the spaces
                pfnn_Z_offsetToCarla = PEDESTRIAN_OFFSET_Z - pfnnPosInCARLASpace.item(2)

                # Convert PFNN bones positions to CARLA space
                for boneId in range(31):
                    bonePos_pfnn    = pfnn_pose[boneId*3 + 0 : boneId*3 + 3]
                    originalBoneZ = pfnn_pose[boneId*3 + 1] # 1 because source has Y as UP axis
                    bonePos_model   = agentPosFromPFNNToModel(bonePos_pfnn, inverse_rotation_matrix, init_pos)
                    bonePos_carla   = agentPosFromModelToCARLA(bonePos_model, pos_y, centering, R, P, middle, frame)

                    for i in range(3):
                        pfnn_pose[boneId*3 + i] = bonePos_carla.item(i)
                    pfnn_pose[boneId*3 + 2] += pfnn_Z_offsetToCarla + (originalBoneZ * 0.01) # Because in centimeters


                ##################################################################################################
                p = agentPosFromModelToCARLA(agentPosThisFrame_modelSpace, pos_y, centering, R, P, middle, frame)
                # Find agent's next position in CARLA coordinate system

                agent_pose_perFrame.append(pfnn_pose)
                agent_positions.append(p.copy())
                # What direction should the agent be pointing in?
                if frame==0:
                    agent_yaw.append(0)
                    agent_yaw.append(0)
                else:

                    directions.append([p.item(0)-previous_pos[0],p.item(1)-previous_pos[1],0] )
                    directions_angle.append(math.atan2(p.item(1)-previous_pos[1],p.item(0)-previous_pos[0] )*180/math.pi)
                    if False:
                        #len(agent_yaw)>4 and np.sign(directions[-1])==np.sign(directions[-2]) and np.sign(directions[-2])==np.sign(directions[-3]) and np.sign(directions[-3])==np.sign(directions[-4]):
                        # dir=math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi
                        # if len(agent_yaw)>2 and np.sign(dir)==np.sign(agent_yaw[-2]) and  np.sign(-dir)==np.sign(agent_yaw[-1]):
                        #     agent_yaw[-1]=np.mean([dir,agent_yaw[-2]]) # -90)

                        agent_yaw.append(math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi)#math.atan2(p[1]-previous_pos[1],p[0]-previous_pos[0] )*180/math.pi)#-90)
                    else:
                        dir = math.atan2(agent_goal[1]-p[1], agent_goal[0]-p[0]) * 180 / math.pi
                        # if len(agent_yaw) > 2 and np.sign(dir) == np.sign(agent_yaw[-2]) and np.sign(
                        #         -dir) == np.sign(agent_yaw[-1]):
                        #     agent_yaw[-1] = np.mean([dir, agent_yaw[-2]])  # -90)

                        agent_yaw.append(math.atan2(agent_goal[1]-p[1], agent_goal[0]-p[0]) * 180 / math.pi)  # -90)

                    #print str(p[0]-previous_pos[0])+" "+str(p[1]-previous_pos[1])+" "+str(agent_yaw[-1])
                previous_pos=[p.item(0),p.item(1)]
            #agent_yaw.append(yaw_mapping[int(agent_action[frame])]*180)#math.pi)


        # HACK TO SAVE POSES - REMOVE
        posesToSave = [0, 12, 16, 19, 24, 41, 35, 94, 123, 141]
        for poseId in posesToSave:
            agentPoseToSave = agent_pose_perFrame_inPFNNLocal[poseId].copy()
            np.savetxt('pose'+ str(poseId) + '.txt', agentPoseToSave, delimiter=',')  # X is an array
            #copyPose = np.loadtxt('pose0.txt', delimiter=',')
        # END HACK


        camera_yaw=math.atan2(R[1,0],R[0,0])
        camera_pitch = math.atan2(-R[2,0] ,np.sqrt( R[2,1]**2+(R[2,2]**2)))
        camera_roll = math.atan2(R[2,1] , R[2,2])

        actor_list = []
        car_list={}
        pygame.init()

        client = carla.Client('localhost', 2000)
        client.set_timeout(100000.0)

        world = client.get_world()
        print(("initial speed: "+str(np.linalg.norm(directions)*10)))
        control=carla.WalkerControl()
        control.speed=np.linalg.norm(directions)*10

        print('enabling synchronous mode.')
        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)


        cur_yaw=0
        name_movie=""
        name_movie_main=""
        try:
            m = world.get_map()
            print ("starting ")
            poses= m.get_spawn_points()

            start_pose=poses[int(nbr)]

            print ("Location")
            print((start_pose.location))
            print (middle_2)
            print (middle_w)
            print ("Old Location")
            print((start_poses[int(nbr)]))
            print ("Centering")
            print((centering["middle"]))
            init_pos = carla.Location(x=middle_w[0,0], y=middle_w[1,0], z=PEDESTRIAN_OFFSET_Z)
            init_rot = carla.Rotation(yaw=0, pitch=0, roll=0)
            blueprint_library = world.get_blueprint_library()
            vehicles=blueprint_library.filter('vehicle.*')
            car_bp = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 4]

            if not actor_perspective:
                init_trans = carla.Transform(init_pos, init_rot)
                spectator = world.get_spectator()
                spectator.set_transform(init_trans)

            print ("Position of actor")
            print((agent_positions[0]))

            agent_init_pos = carla.Location(x=agent_positions[0].item(0), y=agent_positions[0].item(1), z=PEDESTRIAN_OFFSET_Z)
            print (agent_init_pos)
            agent_init_rot = carla.Rotation(yaw=directions_angle[0], pitch=0, roll=0)
            print (agent_init_rot)

            walkersBPs = blueprint_library.filter(PEDESTRIAN_BLUEPRINT_FOR_MAIN_ACTOR)
            bp = random.choice(walkersBPs)



            transform =  carla.Transform(agent_init_pos, agent_init_rot)
            print ("Here")
            #if not actor_perspective:
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                print ("Initialized actor")
            else:
                "Failed to initialize actor"
            # else:
            #     actor=None
            #     spectator = world.get_spectator()
            #     spectator.set_transform(transform)
            #static.prop.box03


            # Place out goal in CARLA coordinates
            rotation = actor.get_transform().rotation
            #actor.set_velocity(carla.Vector3D(directions[0][0]/np.linalg.norm(disp),directions[0][1]/np.linalg.norm(disp),0.0 ))
            goal_box = None
            goal_based=True
            if goal_based:
                bp_goal = random.choice(blueprint_library.filter('static.prop.box03'))
                agent_goal_pos = carla.Location(x=agent_goal.item(0), y=agent_goal.item(1), z=0.3)
                print (agent_goal_pos)
                agent_goal_rot = carla.Rotation(yaw=0, pitch=0, roll=0)
                print (agent_goal_rot)
                transform_goal = carla.Transform(agent_goal_pos, agent_goal_rot)


                goal_box = world.try_spawn_actor(bp_goal, transform_goal)

            vehicle_map={}
            pedestrian_map={}
            vehicle_vel_map = {}
            pedestrian_vel_map = {}

            prev_pos=[copy.copy(agent_init_pos.x), copy.copy(agent_init_pos.y), copy.copy(agent_init_pos.z)]
            camera_transform = carla.Transform(carla.Location(x=0.3,y=0.0, z=0),carla.Rotation(yaw=0, pitch=0, roll=0))
            if not actor_perspective:
                camera = world.spawn_actor(
                    blueprint_library.find('sensor.camera.rgb'),
                    camera_transform,
                    attach_to=spectator)
            else:
                camera = world.spawn_actor(
                    blueprint_library.find('sensor.camera.rgb'),
                    carla.Transform(carla.Location(x=agent_positions[0].item(0)-5.5, y=agent_positions[0].item(1), z=1+2.8), carla.Rotation(pitch=-15)))
                    #attach_to=actor)

            if DONT_CREATE_OTHER_AGENTS:
                pass
            else:
                # Create cars
                for car_key, car in cars_dict[50].items():
                    npc=None
                    tries=0
                    while npc is None and tries<3:
                        bp=random.choice(car_bp)#blueprint_library.filter('vehicle.*'))#z=1
                        transform=carla.Transform(carla.Location(x=car['transform'][0], y=car['transform'][1], z=0.35),carla.Rotation(yaw=car['yaw'],pitch=0, roll=0))
                        npc = world.try_spawn_actor(bp, transform)
                        tries=tries+1
                        if npc is not None:
                            vehicle_map[car_key]=npc
                            print(('created %s' % car_key))#npc.type_id)

                # Create pedestrians
                ped_controllers={}
                ped_rotations={}
                for car_key, car in people_dict[50].items():
                    npc = None
                    tries = 0
                    while npc is None and tries < 3:
                        bp = random.choice(blueprint_library.filter('walker.pedestrian.0001'))
                        print("Height "+str(car['bounding_box'][2]))
                        # print "Typical yaw"+str(car['yaw'])
                        transform = carla.Transform(carla.Location(x=car['transform'][0], y=car['transform'][1], z=car['bounding_box'][2]+0.3),
                                                    carla.Rotation(yaw=car['yaw'], pitch=0, roll=0))

                        npc = world.try_spawn_actor(bp, transform)
                        tries = tries + 1
                        if npc is not None:
                            pedestrian_map[car_key] = npc
                            ped_controllers[car_key] = carla.WalkerControl()
                            ped_rotations[car_key]=npc.get_transform().rotation
                            print(('created %s' % npc.type_id))


            # Make sync queue for sensor data.
            image_queue = queue.Queue()
            camera.listen(image_queue.put)

            frame = None
            my_frame=0

            display = pygame.display.set_mode(
                (800, 600),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            font = get_font()

            clock = pygame.time.Clock()

            while my_frame<len(agent_positions):
                if my_frame >= FRAME_RECORDING_STOP_FRAME:
                    break

                if should_quit():
                    break

                clock.tick()
                world.tick()
                ts = world.wait_for_tick()

                if frame is not None:
                    if ts.frame_count != frame + 1:
                        logging.warning('frame skip!')

                frame = ts.frame_count

                while True:
                    image = image_queue.get()
                    if image.frame_number == ts.frame_count:
                        break
                    logging.warning(
                        'wrong image time-stampstamp: frame=%d, image.frame=%d',
                        ts.frame_count,
                        image.frame_number)

                # Update car location
                for key, car in vehicle_map.items():
                    if my_frame + 50 < len(cars_dict):
                        if key in cars_dict[my_frame+50]:
                            pos=cars_dict[my_frame+50][key]
                            transform = carla.Transform(
                                carla.Location(x=pos['transform'][0], y=pos['transform'][1], z=0.2),
                                carla.Rotation(yaw=pos['yaw'], pitch=0, roll=0))
                            car.set_transform(transform)

                # Update pedestrian location
                for key, person in pedestrian_map.items():
                    if my_frame+50<len(people_dict):
                        if key in people_dict[my_frame+50]:
                            pos=people_dict[my_frame+50][key]
                            cur_pos = person.get_transform().location
                            disp = [pos['transform'][0] - cur_pos.x,
                                    pos['transform'][1] - cur_pos.y]  # , 1.3 - cur_pos.y]
                            ped_controllers[car_key].speed = np.linalg.norm(disp)  # directions[my_frame])
                            ped_rotations[car_key].yaw=pos['yaw']
                            ped_controllers[car_key].direction = ped_rotations[car_key].get_forward_vector()
                            person.apply_control(ped_controllers[car_key])
                            #angle = math.atan2(disp[1], disp[0]) * 180 / math.pi
                            # transform = carla.Transform(
                            #     carla.Location(x=pos['transform'][0], y=pos['transform'][1], z=pos['bounding_box'][2]+0.3),
                            #     carla.Rotation(yaw=pos['yaw'], pitch=0, roll=0))

                            #person.set_transform(transform)

                # Control the RL- agent's visualization
                if actor is not None:
                    if len(agent_positions)> my_frame:
                        agent_init_pos = carla.Location(x=agent_positions[my_frame].item(0), y=agent_positions[my_frame].item(1), z=PEDESTRIAN_OFFSET_Z)
                        agent_init_rot=carla.Rotation(yaw=agent_yaw[my_frame], pitch=0, roll=0)
                        cur_yaw+=agent_yaw[my_frame]
                        cur_yaw=cur_yaw%360

                        # actor.set_transform(carla.Transform(agent_init_pos, agent_init_rot))
                        camera.set_transform(carla.Transform(carla.Location(x=agent_positions[my_frame].item(0) - 5.5, y=agent_positions[my_frame].item(1), z=1 + 2.8),carla.Rotation(pitch=-15)))



                        cur_pos = actor.get_transform().location
                        disp = [agent_positions[my_frame ].item(0) - cur_pos.x,
                                agent_positions[my_frame ].item(1)- cur_pos.y]#, 1.3 - cur_pos.y]
                        angle = math.atan2(disp[1], disp[0]) * 180 / math.pi
                        print(disp)
                        #if my_frame<5:

                        control.speed =np.linalg.norm(disp) * 0.95 #*1.25#*2.5#*4.5#directions[my_frame])


                        # rotation.yaw = round(rotation.yaw, 1)
                        print("Speed " + str(control.speed))

                        if my_frame < len(directions):
                            print("Direction " + str([directions[my_frame][0], directions[my_frame][1], 0.0]))
                            rotation.yaw =directions_angle[my_frame]
                            rotation.yaw = round(rotation.yaw, 1)

                            # Setting direction problem
                            ###################
                            # This is wrong
                            control.direction = rotation.get_forward_vector()
                            # Could be correct
                            #control.direction = carla.Vector3D(directions[my_frame][0],directions[my_frame][1],0.0 )
                            # Or maybe this
                            #control.direction = carla.Vector3D(disp[0], disp[1], 0.0)


                            control.useForcedTarget = True
                            currentDesired2DPos = agent_positions[my_frame]
                            control.targetPos = carla.Vector3D(currentDesired2DPos.item(0), currentDesired2DPos.item(1), 0.0)
                            control.usePFNN = USE_PFNN_VISUALIZER

                            control.poses = agent_pose_perFrame[my_frame].tolist() #(my_frame + np.arange(31*3)).tolist()
                            #########################


                            print("Frame: " + str(my_frame) + " Agent location "+str(actor.get_transform().location)+" wanted pos "+str(agent_init_pos))

                        # self._rotation.yaw = round(self._rotation.yaw, 1)
                        # self._control.direction = self._rotation.get_forward_vector()
                        actor.apply_control(control)
                        pos_n=actor.get_transform().location
                        new_pos=[prev_pos[0]-pos_n.x, prev_pos[1]-pos_n.y,prev_pos[2]-pos_n.z]
                        print("Actual speed: "+str(np.linalg.norm(new_pos)))
                        prev_pos = [copy.copy(cur_pos.x), copy.copy(cur_pos.y), copy.copy(cur_pos.z)]

                # if actor_perspective:
                #     if len(agent_positions) > my_frame:
                #         agent_init_pos = carla.Location(x=agent_positions[my_frame].item(0),
                #                                         y=agent_positions[my_frame].item(1), z=1.26)
                #         agent_init_rot = carla.Rotation(yaw=cur_yaw, pitch=0, roll=0)
                #         cur_yaw += agent_yaw[my_frame]
                #         cur_yaw = cur_yaw % 360
                #         print "Current yaw "+str( cur_yaw) +" "+str(agent_yaw[my_frame])
                #         spectator.set_transform(carla.Transform(agent_init_pos, agent_init_rot))
                draw_image(display, image)
                my_frame=my_frame+1

                # Where to save images!
                if not actor_perspective:
                    image.save_to_disk(os.path.join(out_path, 'Pictures', 'test_%02d_%02d_%06d.jpg' % (int(nbr),int(epoch),my_frame)))
                    name_movie_main=os.path.join(out_path, 'Pictures', 'test_%02d_%02d' % (int(nbr),int(epoch)))
                    name_movie =name_movie_main+'_%06d.jpg'
                else:
                    image.save_to_disk(os.path.join(out_path, 'Pictures', 'test_perpective_%02d_%02d_%06d.jpg' % (int(nbr), int(epoch), my_frame)))
                    name_movie_main = os.path.join(out_path, 'Pictures', 'test_perpective_%02d_%02d' % (int(nbr),int(epoch)))
                    name_movie = name_movie_main + '_%06d.jpg'
                text_surface = font.render('% 5d FPS' % clock.get_fps(), True, (255, 255, 255))
                display.blit(text_surface, (8, 10))
                pygame.display.flip()

        finally:
            print('\ndisabling synchronous mode.')
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

            print('destroying actors.')

            for key, car in vehicle_map.items():
                car.destroy()

            for key, car in pedestrian_map.items():
                car.destroy()
            camera.destroy()
            if actor is not None:
                actor.destroy()
            pygame.quit()
            if goal_box is not None:
                goal_box.destroy()
            print('done.')
        if not actor_perspective:
            command = "ffmpeg -framerate 10 -i " +name_movie + ' -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + name_movie_main+ '.mp4'
        else:
            command = "ffmpeg -framerate 10 -i " + name_movie + ' -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p -y ' + name_movie_main + '_perspective.mp4'

        print(command)
        subprocess.call(command, shell=True)
