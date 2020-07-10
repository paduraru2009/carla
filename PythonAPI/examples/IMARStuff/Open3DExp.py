import open3d as o3d
import numpy as np
import pytransform3d as py3d
from pytransform3d import rotations
import os
from Open3Dexp_params import *
import time
import subprocess
from enum import Enum
import math
import LineSetCustom
import hashlib
import pickle

#############################################################################################################


# debugging helper to see only selected entities
def filterEntityId(id):
    shownIds = ['Qn1uJcrBAWVxwElg2yVljw', 'tRL_3JCel2dmDFq8Bk343A']
    #if not id in shownIds:
    #    return False
    return True


# PARAMS FOR saving the history of poses
# AND REPLAYING THE POSES FOR VISUALIZATION - TODO REFACTOR THIS
#--------------------------------------------------
g_poseHistory = {} # frame id to pose list of 31 bones

def savePoseHistory(outVisFolder):
    fullPath = os.path.join(outVisFolder, "posesHist.pkl")
    with open(fullPath, "wb") as fileHandle:
        pickle.dump(g_poseHistory, fileHandle)

# TODO: parametrize this + output folders above !!
"""
REPLAY_POSES_HISTORY_LIST = [os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_rgb_traj1, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_rgb_traj2, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_rgb_traj3, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_rgb_traj4, "posesHist.pkl")]


REPLAY_POSES_HISTORY_LIST = [os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_seg_traj1, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_seg_traj2, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_seg_traj3, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE18311_seg_traj4, "posesHist.pkl")]

REPLAY_POSES_HISTORY_LIST = [os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_rgb_traj1, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_rgb_traj2, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_rgb_traj3, "posesHist.pkl")]
                                

REPLAY_POSES_HISTORY_LIST = [os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_seg_traj1, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_seg_traj2, "posesHist.pkl"),
                                os.path.join(OUT_VIS_FOLDER_Waymo_SCENE15646511_seg_traj3, "posesHist.pkl")]    
"""
REPLAY_POSES_HISTORY_LIST = None

REPLAY_MIN_FRAME_ID = 20
REPLAY_MAX_FRAME_ID = 130
REPLAY_FRAMESKIP = 7
#--------------------------------------------------


def setPoseDataToGeometry(poseData, poseGeometries, poseOriginalCoords, YCorrection):
    def setBonePos(boxPoints, P0, P1):
        firstHalf = int(len(boxPoints) / 2)
        for i in range(firstHalf):
            boxPoints[i] = poseOriginalCoords[i] + P0
        for i in range(firstHalf, len(boxPoints)):
            boxPoints[i] = poseOriginalCoords[i] + P1

    geomIter = 0
    for i in range(31):
        if parents[i] == -1:
            continue

        # Get data
        # PFNN is with Y up and Z forward. Sometimes, like in the case of NEEDS_AXIS_INVERSION = False, it means that Z is UP.
        P0 = getBonePosFromPoseData(parents[i], poseData) * POSE_SCALE_FACTOR
        P1 = getBonePosFromPoseData(i, poseData) * POSE_SCALE_FACTOR

        if NEEDS_AXIS_INVERSION:
            P0[1] += YCorrection
            P1[1] += YCorrection
        else:
            P0[2], P0[1] = P0[1], P0[2]
            P1[2], P1[1] = P1[1], P1[2]
            P0[2] += YCorrection
            P1[2] += YCorrection

        # Set data on corresponding geometry
        geom = poseGeometries[geomIter]
        boxPoints = np.asarray(geom.vertices)
        setBonePos(boxPoints, P0, P1)

        geomIter += 1

def getBonePosFromPoseData(boneIndex, poseData):
    return poseData[boneIndex*3 + 0 : boneIndex*3 + 3]

def createOpen3DPoseGeometry(visGeomSet):
    poseGeometries = []
    poseGeometries_originalVertices = []
    colorsCount = SkeletonColors.shape[0]
    colorIndex = 0
    for i in range(31):
        if parents[i] == -1:
            continue

        mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.3, height = 0.3, depth=0.3)
        mesh_box.compute_vertex_normals()
        poseGeometries_originalVertices = np.copy(mesh_box.vertices)

        # Set the color
        #boxColors = np.asarray(mesh_box.colors) # TODO
        thisBoneColor = SkeletonColors[colorIndex]
        colorIndex+= 1
        mesh_box.paint_uniform_color([thisBoneColor[0], thisBoneColor[1], thisBoneColor[2]])

        visGeomSet.add_geometry(mesh_box)
        poseGeometries.append(mesh_box)

    return poseGeometries, poseGeometries_originalVertices

    #mesh_box.translate(P0, relative = True)
    #mesh_box.scale([0.1, size, 0.1], center=False)

def get_pcd_stats(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    center = pcd.get_center()
    bbox_oriented = pcd.get_oriented_bounding_box()
    maxbound = pcd.get_max_bound()
    minbound = pcd.get_min_bound()
    print("Bounding box, max/min: ", maxbound, minbound)

def output_pcd_visualizations():

    '''
    path_aachen = "VideoWork/aachen_000042_copy/meshed.ply"
    path_base = "VideoWork/reconstructions/meshed.ply"
    path_base_no_ext = path_base[:path_base.find('.')]
    paths = [path_aachen]
    for i in range(-1, 8):
        path_input = path_base_no_ext + " (" + str(i) + ").ply"
        path_out = "VideoWork/reconstructions/Vis/" + os.path.basename(path_input) + ".png"
        customDrawPointCloud(path_input, path_out)
    '''

    paths=["VideoWork/reconstructions/weimar_000021/meshed.ply",
            "VideoWork/reconstructions/tubingen_000136/meshed.ply",
            "VideoWork/reconstructions/tubingen_000112/meshed.ply",
            "VideoWork/reconstructions/bremen_000028/meshed.ply",
            "VideoWork/reconstructions/weimar_000013/meshed.ply",
            "VideoWork/reconstructions/bremen_000284/meshed.ply",
            "VideoWork/reconstructions/weimar_000094/meshed.ply",
            "VideoWork/reconstructions/bremen_000160/meshed.ply",
            "VideoWork/reconstructions/bremen_000227/meshed.ply",
            "VideoWork/reconstructions/darmstadt_000019/meshed.ply",
            "VideoWork/reconstructions/munster_000039/meshed.ply"]
            #"VideoWork/reconstructions/stuttgart_000154/meshed.ply"]

    for path_input in paths:
        pos1 = path_input.rfind('/')
        pos2 = path_input[:pos1].rfind('/')
        print("doing file " + path_input)
        path_out = "VideoWork/Vis/" + path_input[pos2 + 1 : pos1] + ".png"
        customDrawPointCloud(path_input, path_out)


################################################################################################

# TODO use speed per point
deltaTime_at60fps = 0.0167 # THis is the 'normal' update time to give you 60 fps
deltaTime_inMySimulation = 0.0333 # Let's say you have 30 fps in your simulation
INVALID_TRAJECTORY_INDEX = -1

def distance2D(x0, z0, x1, z1):
    return math.sqrt((x0-x1)**2 + (z0-z1)**2)

class SimulationType(Enum):
    PFNN_SIMULATION = 0
    DEBUG_SIMULATION = 1 # Take data from some file

def readPCDFromPath(path, frameId = -1, useSegView = False, allowToFail = False):
    fullPath = None
    if frameId == -1:
        fullPath = path
    else:
        fileStrFormat = "combined_carla_moving_segColor_f{0:05d}_conv.ply" if useSegView == True else "combined_carla_moving_f{0:05d}_conv.ply"
        fullPath = os.path.join(path, fileStrFormat.format(frameId))

    doesFileExists = (os.path.exists(fullPath) and os.path.isfile(fullPath))
    assert allowToFail == True or doesFileExists == True

    if not doesFileExists:
        return None

    # Read point cloud data
    pcd = o3d.io.read_point_cloud(fullPath)
    #get_pcd_stats(pcd)
    pcd = pcd.rotate(R, center=np.array([0, 0, 0]))
    pcd = pcd.scale(SCALE_FACTOR, center=np.array([0, 0, 0]))
    # pcd = pcd.translate([0, 0, 0])
    # pcd = pcd.voxel_down_sample(voxel_size=10)
    #get_pcd_stats(pcd)
    return pcd


class SimulationEnv:

    # Simulates a PFNN model running on a given trajectory in a cloud point data
    # simDataPath and transform can be None
    def initEnv(self, cameraFileParams, simData, startPos, simType, outputTransformer = None):
        self.simType = simType
        self.outputTransformer = outputTransformer
        #convertBlenderToPointCloudVisualizer(self.startPos)
        self.cameraFileParams = cameraFileParams
        self.useSegmentationView = simData["USE_SEGMENTATION_VIEW"]
        self.pcd = None

        self.isFixedEnvironment = simData["IS_FIXED_ENVIRONMENT"]
        self.cloudDataPath = simData["EnvironmentPointCloudPath"]
        firstFrameIndex = (-1 if self.isFixedEnvironment == True else simData["START_FRAME_INDEX_ENV"])
        firstPDC = readPCDFromPath(self.cloudDataPath, firstFrameIndex, useSegView=self.useSegmentationView)
        self.pcd = firstPDC

        # Read sim data
        self.simdata_endEnvSimFrame = None
        self.readSimData(simData, R, SCALE_FACTOR_FOR_POSITIONS)

        # Convert start pos
        self.startPos = ((R.dot(startPos)) * SCALE_FACTOR_FOR_POSITIONS) if self.trajectoryNeedsTransform else startPos


        # Create visualizer and set options
        # -------------------------------------
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1628, height=1028)
        self.ctr = self.vis.get_view_control()
        self.parameters = o3d.io.read_pinhole_camera_parameters(self.cameraFileParams)
        self.ctr.convert_from_pinhole_camera_parameters(self.parameters)
        self.prndr = self.vis.get_render_option()
        self.prndr.point_size = POINT_SIZE
        self.prndr.line_width = 50
        self.prndr.background_color = BACKGROUND_COLOR
        # rndr.mesh_show_wireframe = True
        # rndr.mesh_show_back_face = True
        #print(dir(ctr))
        #print(dir(rndr))
        # ctr.rotate(30.0, 0.0)
        # ctr.translate(0, 0, 10000)
        # ctr.set_constant_z_far(300)
        # ctr.set_constant_z_near(0)
        self.ctr.change_field_of_view(step=30)
        # -------------------------------------

        #  SHOW coordinate axis
        if SHOW_COORDINATE_AXIS:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[0, 0, 0])
            centerOfCloud = self.pcd.get_center()
            mesh_frame_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=centerOfCloud)

            self.vis.add_geometry(mesh_frame)
            self.vis.add_geometry(mesh_frame_center)

        # Add point cloud to the geometry
        self.vis.add_geometry(self.pcd)

        # Create pose geometries static (once then update them iteratively)
        # And add them to the visualizer

        self.poseGeometries, self.poseGeometries_originalCoords = createOpen3DPoseGeometry(self.vis)

        if simType == SimulationType.PFNN_SIMULATION:
            from pfnncharacter import PFNNCharacter as Ch
            self.agentInst = Ch("a")
            self.agentInst.init("")

            startPosX_pfnnSystem, startPosY_pfnnSystem = convert2DPosFromPointCloudVisualizerToPFnn(self.startPos[0],
                                                                                                    self.startPos[2] if NEEDS_AXIS_INVERSION is True else self.startPos[1])
            self.agentInst.resetPos(startPosX_pfnnSystem, 0.0, startPosY_pfnnSystem)
            #self.agentInst.setTargetReachedThreshold(TARGET_REACHED_HACK_THRESHOLD)
            self.distanceReachedThreshold = max(self.agentInst.getTargetReachedThreshold(), 100)

            self.YCorrection = self.startPos[1] if NEEDS_AXIS_INVERSION else self.startPos[2]

            print("Agent name: ", self.agentInst.getName())
            print("Agent has: ", self.agentInst.getNumJoints(), " joints")
        elif simType == SimulationType.DEBUG_SIMULATION:
            self.PoseFrameFiles = ["VideoWork/pose0.txt", "VideoWork/pose1.txt", "VideoWork/pose2.txt",
                              "VideoWork/pose3.txt", "VideoWork/pose4.txt", "VideoWork/pose5.txt"]

        #self.frameIndex = simData["START_FRAME_INDEX_ENV"]
        self.savedFrameCounter = 0
        self.nextTrajectoryIndex = INVALID_TRAJECTORY_INDEX
        self.lastPoseData = None


    def readSimData(self, simData, RotationMatrix, ScaleFactor):
        if simData == None:
            return

        # Read data first
        self.simdata_cars = simData["cars"]
        self.simdata_people = simData["people"]
        self.heightOffset = simData["heightOffset"]
        assert (len(self.simdata_cars) == len(self.simdata_people))
        self.simdata_endEnvSimFrame = simData["END_FRAME_INDEX_ENV"] #len(self.simdata_cars)
        self.simdata_frequency = simData["frequency"] # TODO PARAM
        self.simdata_recordFramerate =  simData["recordFramerate"]

        # We want to render the agent at the time specified by agentStartRenderFrame in parallel with environment.
        # Since there is a possible delay between sim and rendering of the agent we might have to start the simulation at negative frame indices
        envStartUpdateFrame = simData["START_FRAME_INDEX_ENV"]
        self.frameIndex = min(envStartUpdateFrame, simData["agentStartSimFrame"])



        self.deltaTime_inMySimulation = 1.0 / self.simdata_frequency
        self.trajectoryNeedsTransform= simData["TRAJECTORY_NEEDS_TRANSFORM"]

        self.agentStartSimFrame = simData['agentStartSimFrame']
        self.agentStartRenderFrame = simData['agentStartRenderFrame']

        self.outVisFolder = simData["SAVE_PARAMS"]["OUT_VIS_FOLDER"]

        def rotScaleTransform(pos, R, S):
            pos = RotationMatrix.dot(pos)
            pos[0] *= ScaleFactor
            pos[1] *= ScaleFactor
            pos[2] *= ScaleFactor
            return pos

        # Convert it to the vizualizer coordinate system
        def convertBlenderSimDataListToPointCloudVizSystem(framedDict):
            for frameItem, frameData in framedDict.items():
                for itemKey, itemData in frameData.items():
                    if not filterEntityId(itemKey):
                        continue
                    bboxMinMax = itemData['BBMinMax']

                    minPos = bboxMinMax[:,0]
                    maxPos = bboxMinMax[:, 1]
                    itemData['BBMinMax'][:, 0] = rotScaleTransform(minPos, RotationMatrix, ScaleFactor)
                    itemData['BBMinMax'][:, 1] = rotScaleTransform(maxPos, RotationMatrix, ScaleFactor)
                    itemData['BBMinMax'][2, :] -= self.heightOffset
                    minPos = minPos

        convertBlenderSimDataListToPointCloudVizSystem(self.simdata_cars)
        convertBlenderSimDataListToPointCloudVizSystem(self.simdata_people)

        self.peopleGeom = []
        self.carsGeom = []

    # Given either people or cars sim data, we interpolate and create the scene geometries for them then return
    def createSimEntitiesFromData(self, simDataEntities, frameIndex, colorsList):
        geomResult = []
        simFrame = frameIndex #/ self.simdata_frequency
        #betweenFramesPercent = (simFrame % self.simdata_frequency) / self.simdata_frequency
        simDataEntities_f0 = simDataEntities[simFrame]
        #simDataEntities_f1 = simDataEntities[simFrame + 1] if simFrame + 1 < self.simdata_numFrames else simDataEntities_f0

        for entityKey, entityData in simDataEntities_f0.items():
            if not filterEntityId(entityKey):
                continue

            entityBBox = entityData["BBMinMax"]
            entityMin = entityBBox[:, 0]
            entityMax = entityBBox[:, 1]

            entityGeom = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector([entityMin, entityMax]))
            entityGeom.color = colorsList[hash(entityKey) % len(colorsList)]
            entityGeom.scale(scale=INV_SCALE_FACTOR, center=np.array([0,0,0]))
            geomResult.append(entityGeom)

        return geomResult

    # Adds, remove, simulate other info for environments such as cars and pedestrians
    # Giving the simFrame and the time to interpolate between in the given frame to the next
    def updateEnvironment(self, simFrame):
        # The simulation has not started yet
        if simFrame <= 0:
            return True

        # If we target a max number of frames stop the simulation after
        if self.simdata_endEnvSimFrame == None or self.frameIndex >= self.simdata_endEnvSimFrame:
            return False

        # Should i load a new point cloud every frame ?
        if self.isFixedEnvironment == False:
            framePCD = readPCDFromPath(self.cloudDataPath, self.frameIndex, useSegView=self.useSegmentationView, allowToFail=True)
            if framePCD == None:
                return False# No more cloud file for simulation ? exit
            
            self.vis.remove_geometry(self.pcd)
            self.pcd = framePCD

            self.vis.add_geometry(self.pcd)
            self.ctr = self.vis.get_view_control()
            self.parameters = o3d.io.read_pinhole_camera_parameters(self.cameraFileParams)
            self.ctr.convert_from_pinhole_camera_parameters(self.parameters)

        # Remove the previously created geometry and create/add the new ones
        # We do this because their number is very different from frame to frame and there is no much optimization opportunity..
        for geom in self.peopleGeom:
            self.vis.remove_geometry(geom, reset_bounding_box=False)

        for geom in self.carsGeom:
            self.vis.remove_geometry(geom, reset_bounding_box=False)

        # BBOX Sizes from citiscapes - TODO: move them in processed data as bboxes similar to waymo now: cars np.array([40, 15, 40], np.array([20, 25, 20]

        self.carsGeom   = self.createSimEntitiesFromData(self.simdata_cars, self.frameIndex, Cars_Colors)
        self.peopleGeom = self.createSimEntitiesFromData(self.simdata_people, self.frameIndex, Pedestrians_Colors)

        for geom in self.carsGeom:
            self.vis.add_geometry(geom, reset_bounding_box=False)
        for geom in self.peopleGeom:
            self.vis.add_geometry(geom, reset_bounding_box=False)

        return True

    # If given, recordingZPosStart will start recording from that Z forward
    def simulatePFNNOnCloudDataTrajectory(self, trajectory, speeds, recordingZPosStart = None, recordingFrameIndex = None, save = False):
        global SAVE_POSE_HISTORY # TODO : put these as parameters...
        global MAX_POSES_IN_HIST
        global g_poseHistory

        # Transform trajectory points to data cloud coordinate reference
        self.fixedTrajectory = trajectory.copy()
        self.fixedSpeeds = speeds.copy()

        if self.trajectoryNeedsTransform == True:
            transformTrajectoryPointsFromBlenderToPointCloudVis(self.fixedTrajectory)

        # Render trajectory points for debugging or visualization
        if SHOW_TRAJECTORY_DEBUG:
            # Again, these should be enabled only for image visualization debugging
            shouldShowMultipleTrajectories = (SHOW_TRAJECTORIES_FOR_VIS is not None and len(SHOW_TRAJECTORIES_FOR_VIS) > 0)
            if shouldShowMultipleTrajectories:
                trajectoriesToShow = SHOW_TRAJECTORIES_FOR_VIS
                trajectoriesToShow_colors = SHOW_TRAJECTORIES_FOR_VIS_COLORS
            else:
                trajectoriesToShow = [self.fixedTrajectory] # Showing only the current trajectory path
                trajectoriesToShow_colors = [1.0, 1.0, 0.0]

                # Trajectories waypoints
                prevPos = None
                for trajIdx, T in enumerate(trajectoriesToShow):
                    for pointIdx, pos3d in enumerate(T):
                        mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
                        mesh_box.translate(pos3d)
                        self.vis.add_geometry(mesh_box)

                        pointsList = trajectoriesToShow[trajIdx] if shouldShowMultipleTrajectories else trajectoriesToShow[trajIdx][1:] # because the first point is a bit faked to allow the agent to move a bit
                        line_set = LineSetCustom.LineMesh(points=pointsList, lines=None,
                                                            colors=trajectoriesToShow_colors, radius=0.25)
                        for cylinder in line_set.cylinder_segments:
                            self.vis.add_geometry(cylinder)

        # Should i show the goal ?
        if SHOW_GOAL_DEBUG:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
            mesh_sphere.translate(self.fixedTrajectory[-1]) # Last point on the trajectory
            self.vis.add_geometry(mesh_sphere)

        # Set camera params again !
        self.ctr = self.vis.get_view_control()
        self.parameters = o3d.io.read_pinhole_camera_parameters(self.cameraFileParams)
        self.ctr.convert_from_pinhole_camera_parameters(self.parameters)

        #if outPath == None:
        #    vis.run()
        self.updateEnvironment(self.frameIndex)


        # Special case to add several geometries from HISTORY list for image visualization
        # SHould be null in general !!
        global REPLAY_POSES_HISTORY_LIST
        if REPLAY_POSES_HISTORY_LIST is not None and len(REPLAY_POSES_HISTORY_LIST) > 0:
            # First read all poses
            for posesPath in REPLAY_POSES_HISTORY_LIST:
                with open(posesPath, "rb") as fileHandle:
                    posesDict = pickle.load(fileHandle)
                    for key,value in posesDict.items():
                        if key not in g_poseHistory:
                            g_poseHistory[key] = []
                        g_poseHistory[key].append(np.array(value))

            # For each frame data
            for pose_frameIndex, poseListData in g_poseHistory.items():
                if REPLAY_MIN_FRAME_ID > pose_frameIndex or pose_frameIndex > REPLAY_MAX_FRAME_ID:
                    continue
                if pose_frameIndex % REPLAY_FRAMESKIP != 0:
                    continue

                # For each pose data saved on this frame
                for poseData in poseListData:
                    # Create its geometry (added to the visualizer as well)
                    self.poseGeometries, self.poseGeometries_originalCoords = createOpen3DPoseGeometry(self.vis)

                    setPoseDataToGeometry(poseData, self.poseGeometries, self.poseGeometries_originalCoords, -self.heightOffset)

                    # Update visualization to reflect the new posed data
                    for poseGeom in self.poseGeometries:
                        self.vis.update_geometry(poseGeom)

        self.ctr = self.vis.get_view_control()
        self.parameters = o3d.io.read_pinhole_camera_parameters(self.cameraFileParams)
        self.ctr.convert_from_pinhole_camera_parameters(self.parameters)
        ############################################################################

        self.vis.run()

        isSimOver = not VIEW_ENVIRONMENT_SIMULATION == True
        while not isSimOver:
            print("----- Sim Frame ", self.frameIndex)

            # Sometimes for faster execution we might disable the environment
            if IS_ENVIRONMENT_UPDATING_ENABLED:
                res = self.updateEnvironment(self.frameIndex)
            else:
                res = True

            if IS_PFNN_ENABLED:
                if self.frameIndex >= self.agentStartSimFrame:
                    isSimOver = isSimOver or self.simulateFrame()
                    poseData = self.getNextPoseFromStream()

                if self.frameIndex >= max(self.agentStartRenderFrame, self.agentStartSimFrame):
                    # BEGIN POSE HACK
                    setPoseDataToGeometry(poseData, self.poseGeometries, self.poseGeometries_originalCoords, self.YCorrection)

                    if SAVE_POSE_HISTORY:
                        g_poseHistory[self.frameIndex] = list(poseData.copy())
                    # END POSE HACK

                for poseGeom in self.poseGeometries:
                    self.vis.update_geometry(poseGeom)

            self.vis.poll_events()
            self.vis.update_renderer()

            if save == True:
                if (recordingZPosStart != None and np.asarray(self.poseGeometries[0].vertices)[0][2] < recordingZPosStart) or (recordingFrameIndex <= self.frameIndex):
                    outPath = os.path.join(self.outVisFolder, "frame_{0:05d}.png".format(self.savedFrameCounter))
                    self.vis.capture_screen_image(outPath, True)

                    if self.outputTransformer:
                        self.outputTransformer.transform(outPath, outPath)
                    self.savedFrameCounter += 1

            time.sleep(self.deltaTime_inMySimulation)
            self.frameIndex += 1

            if res == False and (IS_PFNN_ENABLED == False or STOP_PFNN_WHEN_ENVIRONMENT_UPDATE_ENDS == True):
                isSimOver = True

            if  SAVE_POSE_HISTORY and (isSimOver or self.frameIndex >= (5 + MAX_POSES_IN_HIST)):
                savePoseHistory(self.outVisFolder)
                SAVE_POSE_HISTORY = False
                exit(0)


        self.vis.destroy_window()
        if save == True:
            #command = "ffmpeg -framerate 10 -i" +" 'frame_%05d.png'" + " -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p VideoWork/Output/out.mp4"
            command = "ffmpeg -y -framerate " + str(self.simdata_recordFramerate) + " -i " + self.outVisFolder + '/frame_%05d.png' + " -c:v libx264 -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 -pix_fmt yuv420p VideoWork/Output/out.mp4"
            print(command)
            subprocess.call(command, shell=True)

    # Get the pose corresponding to the last simulated step
    def getNextPoseFromStream(self):
        if self.simType == SimulationType.PFNN_SIMULATION:
            self.lastPoseData = self.agentInst.getCurrentPose()

        elif self.simType == SimulationType.DEBUG_SIMULATION:
            if self.frameIndex < len(self.PoseFrameFiles):
                fileName = self.PoseFrameFiles[self.frameIndex]
                self.lastPoseData = np.loadtxt(fileName, delimiter=',')

        return self.lastPoseData

    # Returns False if simulation should continue to run
    def simulateFrame(self):
        isSimFinished = False

        if self.simType == SimulationType.PFNN_SIMULATION:


            # Check first if we need to change our next target point index
            #-----------------------------------------------------------------
            agentX_pfnnSystem, agentZ_pfnnSystem = self.agentInst.getCurrent2DPos()
            agentX, agentZ = convert2DPosFromPFnnToPointCloudVisualizer(agentX_pfnnSystem, agentZ_pfnnSystem)
            DEBUG_LOG("Current Agent pos: in PFNN system: {0:02f},{1:02f}. in PointCloud vizualizer system pos: {2:02f},{3:02f}".format(agentX_pfnnSystem, agentZ_pfnnSystem, agentX, agentZ))

            # NOTE: IF NEEDS_AXIS_INVERSION is False, names containing Z below are actually Y !  Z is up in this case, Y forward!!
            # However, this is convinient since in PFNN Z is forward, Y being up..
            targetX_inPFNNSystem, targetZ_inPFNNSystem = None, None
            if self.nextTrajectoryIndex != INVALID_TRAJECTORY_INDEX:
                targetPos = self.fixedTrajectory[self.nextTrajectoryIndex]
                targetX_inPFNNSystem, targetZ_inPFNNSystem = convert2DPosFromPointCloudVisualizerToPFnn(targetPos[0],
                                                                                                        targetPos[2] if NEEDS_AXIS_INVERSION == True else targetPos[1])

            # If we don't have a target selected yet or we are close to our next target location, then choose the next one
            actualDistance = None if (targetX_inPFNNSystem == None or targetZ_inPFNNSystem == None) else distance2D(agentX_pfnnSystem, agentZ_pfnnSystem, targetX_inPFNNSystem, targetZ_inPFNNSystem)
            if actualDistance == None or actualDistance < self.distanceReachedThreshold:
                self.nextTrajectoryIndex += 1
                if self.nextTrajectoryIndex == len(self.fixedTrajectory):
                    return True


                targetPos = self.fixedTrajectory[self.nextTrajectoryIndex]
                # HACK - MOVE TARGET FORWARD BUT IN THE SAME MOVEMENT DIRECTION BECAUSE OF A PFNN BUG :(
                targetPos_X = (targetPos[0] - agentX)*100.0 + agentX
                targetPos_Z = ((targetPos[2] if NEEDS_AXIS_INVERSION == True else targetPos[1]) - agentZ)*100.0 + agentZ

                targetX_inPFNNSystem, targetZ_inPFNNSystem = convert2DPosFromPointCloudVisualizerToPFnn(targetPos_X, targetPos_Z)
                self.agentInst.setTargetPosition(targetX_inPFNNSystem, targetZ_inPFNNSystem)
                DEBUG_LOG("## New target set for agent. PFNN system: {0:02f},{1:02f}. PointCloud vizualizer system pos: {2:02f},{3:02f}".format(targetX_inPFNNSystem, targetZ_inPFNNSystem, targetPos_X, targetPos_Z))

                # Set agent speed too !
                self.agentInst.setDesiredSpeed(self.fixedSpeeds[self.nextTrajectoryIndex])

            #------------------------------------------------------------------------------

            maxSpeedReached = 0
            enableVerboseLogging = False

            remainingDeltaTime = self.deltaTime_inMySimulation
            while remainingDeltaTime > 0:
                # Update one tick for you environment. Your frame could be split in many small parts to update the simulation correctly
                # Update and post update a simulation frame with the given time
                deltaTime = min(deltaTime_at60fps, remainingDeltaTime)
                remainingDeltaTime -= deltaTime

                self.agentInst.updateAnim(deltaTime)
                self.agentInst.postUpdateAnim(deltaTime)

                # Update the maximum speed obtained so far
                speed = self.agentInst.getCurrentAvgSpeed()
                if maxSpeedReached < speed:
                    maxSpeedReached = speed

                # Get Current position in 2D space
                x, z = self.agentInst.getCurrent2DPos()

            # Printing one of the bones. You have access to pose on each frame ! Proving just for a single bone
            poseData = self.agentInst.getCurrentPose()

            if enableVerboseLogging:
                print(
                    "------ \nFrame: {0}. Deltatime given: {1:.2f}".format(self.frameIndex, deltaTime_inMySimulation))
                print("New agent pos in 2D: x={0:.2f}, z={1:.2f}. AvgSpeed={2:.2f}".format(x, z, speed))

            '''
            # Update one tick
            remainingDeltaTime = deltaTime_inMySimulation
            #speedInThisTick = 0
            while remainingDeltaTime > 0:
                deltaTime = min(deltaTime_at60fps, remainingDeltaTime)
                remainingDeltaTime -= deltaTime

                # Update and post update a simulation frame with the given time
                self.agentInst.updateAnim(deltaTime)
                self.agentInst.postUpdateAnim(deltaTime)

                # Update the maximum speed obtained so far
                #speed = self.agentInst.getCurrentAvgSpeed()
                #if self.maxSpeedReached < speed:
                #    self.maxSpeedReached = speed
                # Since there can be multiple speeds in a single simulation tick, take the maximum one
                #if self.speedInThisTick < speed:
                #    self.speedInThisTick = speed

            speed = self.agentInst.getCurrentAvgSpeed()
            DEBUG_LOG("Agent Speed: {0:02f}".format(speed))
            '''
            return False

        elif self.simType == SimulationType.DEBUG_SIMULATION:
            return self.frameIndex < len(self.PoseFrameFiles)

import cv2
class ImgTransformer:
    def __init__(self, cropX, cropY, cropW, cropH, scaleFactor):
        self.cropX, self.cropY, self.cropW, self.cropH, self.scaleFactor = cropX, cropY, cropW, cropH, scaleFactor

    def transform(self, inputPath, outputPath):
        img = cv2.imread(inputPath)
        img = img[self.cropY: self.cropY + self.cropH, self.cropX: self.cropX + self.cropW]
        img = cv2.resize(img, (int(self.cropW * self.scaleFactor), int(self.cropH * self.scaleFactor)))
        cv2.imwrite(outputPath, img)

def main():

    # TODO: send these as parameters

    ################# DATA FOR AACHEN PATHS #####################
    '''
    trajectory = trajectory_aachen000042_1
    speeds = speeds_aachen000042_1
    startPos = startPos_aachen000042_1
    '''

    '''
    trajectory = trajectory_aachen000042_2
    speeds = speeds_aachen000042_2
    startPos = startPos_aachen000042_2
    '''

    '''
    trajectory = trajectory_aachen000042_3
    speeds = speeds_aachen000042_3
    startPos = startPos_aachen000042_3
    '''
    #recordingZPosStart_aachen00042 = -trajectory[0][1]
    #outTransformer_aachen000042 = ImgTransformer(cropX=511, cropY=428, cropW=561, cropH=486, scaleFactor=2.0)
    ################################################################

    SIMULATE_TUBINGEN = False
    SIMULATE_WAYMO = True

    if SIMULATE_TUBINGEN:
        ################# DATA FOR TUBINGEN 000112 PATHS ######################
        pointCloudDataPath = pointCloudDataPath_Tubingen_000112
        startPos = startPos_tubingen
        trajectory = trajectory_tubingen_000112_2
        speeds = speeds_tubingen_000112_2

        simData = simData_tubingen # Can be NONE
        outTransformer_tubingen = ImgTransformer(cropX=487, cropY=425, cropW=1001, cropH=533, scaleFactor=2.0)
        ################################################################

        outTransformer = outTransformer_tubingen

        simEnv = SimulationEnv()
        simEnv.initEnv(CAMERA_PARAMS_FILE_TUBINGEN, pointCloudDataPath, simData, startPos, SimulationType.PFNN_SIMULATION, outTransformer)
        simEnv.simulatePFNNOnCloudDataTrajectory(trajectory, speeds, recordingZPosStart = None, recordingFrameIndex = 0, save = False)
    elif SIMULATE_WAYMO:
        simData = simData_Waymo

        startPos = simData["SIM_AGENT_START_POS"]
        trajectory = simData["SIM_AGENT_TRAJECTORY"]
        speeds = simData["SIM_AGENT_SPEEDS"]

        outTransformer_Waymo = ImgTransformer(cropX=simData["SAVE_PARAMS"]["CROP_X"],
                                              cropY=simData["SAVE_PARAMS"]["CROP_Y"],
                                              cropW=simData["SAVE_PARAMS"]["CROP_WIDTH"],
                                              cropH=simData["SAVE_PARAMS"]["CROP_HEIGHT"],
                                              scaleFactor=simData["SAVE_PARAMS"]["SCALE_FACTOR"]) # TODO put these on params file
        ################################################################

        outTransformer = outTransformer_Waymo

        simEnv = SimulationEnv()
        simEnv.initEnv(simData["CameraFile"], simData, startPos, SimulationType.PFNN_SIMULATION, outTransformer)
        simEnv.simulatePFNNOnCloudDataTrajectory(trajectory, speeds, recordingZPosStart = None, recordingFrameIndex = 0, save = simData["SAVE_PARAMS"]["ENABLED"])


if __name__== "__main__":
    main()
