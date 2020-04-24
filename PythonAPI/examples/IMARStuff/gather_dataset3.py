"""
Requirements:
 0.    # Here we will run a single episode with 300 frames.
      number_of_frames = 30
      frame_step = 10  # Save one image every 100 frames

 1. 3 types of camera: RGB, Seg, Depth

    image_size = [800, 600]
    camera_local_pos = [0.3, 0.0, 1.3] # [X, Y, Z]
    camera_local_rotation = [0, 0, 0]  # [pitch(Y), yaw(Z), roll(X)]
    fov = 70????????? can we control it or should be 90 ???

 2. Env settings:

     // We start 150 episodes from different spawn points.
     for start_pos in range(150)
            // Output defnition
          //-----------------------------------------------------------------------------------------
          // On each episode we save in OUtput stuff in:  /_out/pos_${startPos} these files using pickle
          // A camera intrisics, canera duct definition,
          // the cars and people details in details.


            A. Intricics  The (3, 3) K Matrix
            K = np.identity(3)
            K[0, 2] = image_size[0]/ 2.0
            K[1, 2] = image_size[1] / 2.0
            K[0, 0] = K[1, 1] = image_size[0] / (2.0 * np.tan(fov * np.pi / 360.0))
            with open(output_folder + '/camera_intrinsics.p', 'w') as camfile:
                pickle.dump(K, camfile)



          //-----------------------------------------------------------------------------------------

            // Step 1: Start a new episode at current start point index
             settings.set(
                        SynchronousMode=True,
                        SendNonPlayerAgentsInfo=True,
                        NumberOfVehicles=100,
                        NumberOfPedestrians=500,
                        WeatherId=random.choice([1, 3, 7, 8, 14]))
            On Each episode settings.randomize_seeds()
            client.load_settings(settings)

            Create the 3 camera sensors: depth, rgb, seg

            # Start at location index id '0'
            client.start_episode(start_i)

            # Capture a few frames
            for frame in range(1, number_of_frames):
                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                 # Save one image every 'frame_step' frames
                if not frame % frame_step:
                    continue

                Save each sensor_data in  filename ='{:s}/{:0>6d}'.format( sensor_name, frame)

                # Get the point cloud colored in segmentation and RGB format
                # Save them as {0>6d).format(frameIdx).ply  and frameIdx_seg.ply
                #TODO: the old version was doing this to transform both the pixels in RGB and Seg to colored 3D point clouds
                # A. PointsInFilmSpace = Transform using using depth the points from 2D image space to camera space  (K^-1 * Pimg).
                # B. PointsInCameraSpace = PointsInFilmSpace  *  Depths for pixels // Now the points are in the camera space, 3D, colored by input given
                # C. get C1=CameraToCar transform, and C2=CarToWorld.   PointsInWorld3DSpace = C1*C2*PointsInCameraSpace # Now we have them all colored in 3D space, the real one !
                # D. Save as ply files
"""


from __future__ import print_function

import glob
import os
import sys
import random
import logging
import time
import argparse
import re

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import copy
import numpy as np
import pickle
import pygame
import traceback
import shutil
from enum import Enum

try:
    import queue
except ImportError:
    import Queue as queue

def check_far(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(
            "{} must be a float between 0.0 and 1.0")
    return fvalue

SYNC_TIME = 0.1
SYNC_TIME_PLUS = 3.0 # When destroying the environment

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def dot3D(v1, v2):
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)

class RenderUtils(object):
    class EventType(Enum):
        EV_NONE = 0
        EV_QUIT = 1
        EV_SWITCH_TO_DEPTH = 2
        EV_SWITCH_TO_RGBANDSEG = 3

    @staticmethod
    def draw_image(surface, image, blend=False):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    @staticmethod
    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    @staticmethod
    def get_input_event():
        for event in pygame.event.get():
            # See QUIT first then swap
            if event.type == pygame.QUIT:
                return RenderUtils.EventType.EV_QUIT
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return RenderUtils.EventType.EV_QUIT
                if event.key == pygame.K_d:
                    return RenderUtils.EventType.EV_SWITCH_TO_DEPTH
                if event.key == pygame.K_s:
                    return RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
        return RenderUtils.EventType.EV_NONE

# Manager to synchronize output from different sensors.

class SensorsDataManagement(object):
    def __init__(self, world, fps, sensorsDict):
        self.world = world
        self.sensors = sensorsDict  # [name->obj instance]
        self.frame = None

        self.delta_seconds = 1.0 / fps
        self._queues = {}  # Data queues for each sensor given + update messages
        self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG

        def make_queue(register_event, sensorname):
            q = queue.Queue()
            register_event(q.put)
            assert self._queues.get(sensorname) is None
            self._queues[sensorname] = q

        make_queue(self.world.on_tick, "worldSnapshot") # The ontick register event
        for name, inst in self.sensors.items():
            make_queue(inst.listen, name)

    # Tick the manager using a timeout to get data and a targetFrame that the parent is looking to receive data for
    def tick(self, targetFrame, timeout):
        data = {name: self._retrieve_data(targetFrame, q, timeout) for name, q in self._queues.items()}
        assert all(inst.frame == targetFrame for key,inst in data.items())  # Need to get only data for the target frame requested

        """
        print("Debug print the queue status")
        for name,inst in self._queues.items():
            print(f"queue name {name} has size {inst.qsize()}")
        """
        return data

    # Gets the target frame from a sensor queue
    def _retrieve_data(self, targetFrame,  sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            # assert data.frame <= targetFrame, ("You are requesting an old frame which was already processed !. data %d, target %d" % (data.frame, targetFrame))
            if data.frame == targetFrame:
                return data

import math


class DataCollector(object):

    # Here we put static things about the environment, simulation , params etc.
    class EnvSettings:

        # Capture params
        maxNumberOfEpisodes = 3 # On each episode it will be spawned on a new data point
        number_of_frames_to_capture = 100 # Number of frames to capture in total from an episode
        frame_step = 10  # Save one image every 10 frames

        # Sim params
        fixedFPS = 30 # FPS for recording and simulation
        tm_port = 8000

        # How to find the episode LENGTH
        # Explanation: In one second you have fixedFPS. We record a measure at each frame_step frames. If fixedFPS=30
        # then in one second you have 3 captures.
        # If number_of_frames_to_capture = 200 => An episode length is 100/3 seconds

        # Camera specifications, size, local pos and rotation to car
        image_size = [800, 600]
        #camera_local_pos = carla.Location(x=0.3, y=0.0, z=1.3)  # [X, Y, Z] of local camera
        camera_front_transform = carla.Transform(carla.Location(x=0.8, z=1.65))
        fov = 70
        isAutopilotForPlayerVehicle = False

        # Environment settings
        MapsToTest = ["Town03"]
        NumVehicles = 20
        NumPedestrians = 30
        vehicles_filter_str = "vehicle.*"
        walkers_filter_str = "walker.pedestrian.*"


        # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations as start/destination points
        PedestriansSpawnPointsFactor = 100
        PedestriansDistanceBetweenSpawnpoints = 4 # m

        OUTPUT_DATA_PREFIX = "out/%s/episode_%d_%d"  # out/MapName/episode_index_spawnPointIndex
        OUTPUT_SEG = "CameraSeg"
        OUTPUT_SEGCITIES = "CameraSegCities"
        OUTPUT_DEPTH = "CameraDepth"
        OUTPUT_DEPTHLOG = "CameraDepthProc"
        OUTPUT_RGB = "CameraRGB"
        TIMEOUT_VALUE = 10.0
        ENABLED_SAVING = False # If activated, output saving data is enabled
        CLEAN_PREVIOUS_DATA = True # If activated, previous data folder is deleted

        # If true, then car stays and waits for pedestrians and other cars to move around. If
        # If false, then on each frame it moves to a random proximity waypoint
        STATIC_CAR = True

        @staticmethod
        def getOutputFolder_seg(baseFolder):
            return os.path.join(baseFolder, DataCollector.EnvSettings.OUTPUT_SEG)

        @staticmethod
        def getOutputFolder_segcities(baseFolder):
            return os.path.join(baseFolder, DataCollector.EnvSettings.OUTPUT_SEGCITIES)

        @staticmethod
        def getOutputFolder_rgb(baseFolder):
            return os.path.join(baseFolder, DataCollector.EnvSettings.OUTPUT_RGB)

        @staticmethod
        def getOutputFolder_depth(baseFolder):
            return os.path.join(baseFolder, DataCollector.EnvSettings.OUTPUT_DEPTH)

        @staticmethod
        def getOutputFolder_depthLog(baseFolder):
            return os.path.join(baseFolder, DataCollector.EnvSettings.OUTPUT_DEPTHLOG)

        @staticmethod
        def configureCameraBlueprint(cameraBp):
            cameraBp.set_attribute('image_size_x', str(DataCollector.EnvSettings.image_size[0]))
            cameraBp.set_attribute('image_size_y', str(DataCollector.EnvSettings.image_size[1]))
            cameraBp.set_attribute('fov', str(DataCollector.EnvSettings.fov))
            # Set the time in seconds between sensor captures
            #cameraBp.set_attribute('sensor_tick', '1.0')


    # Get the weather presets lists
    def find_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


    # Promote uniform sampling around map + spawn in front of walkers
    def uniformSampleSpawnPoints(self, allSpawnpoints, numToSelect):
        availablePoints = [(index, transform) for index, transform in allSpawnpoints]
        selectedPointsAndIndices = [] #[None]*numToSelect

        for selIndex in range(numToSelect):
            # Select the one available that is furthest from existing ones
            bestPoint = None
            bestDist = -1
            for x in availablePoints:
                target_index = x[0]
                target_transform = x[1].location

                # Find the closest selected point to x
                closestDist = math.inf
                closestSelPoint = None
                for y in selectedPointsAndIndices:
                    selPointLocation = y[1].location
                    d = compute_distance(target_transform, selPointLocation)
                    if d < closestDist:
                        closestDist     = d
                        closestSelPoint = y

                if closestSelPoint == None or bestDist < closestDist:
                    bestDist = closestDist
                    bestPoint = x

            if  bestPoint != None:
                availablePoints.remove(bestPoint)
                selectedPointsAndIndices.append(bestPoint)

        return selectedPointsAndIndices

    def prepareOutputFolder(self, folderPath):
        if self.EnvSettings.CLEAN_PREVIOUS_DATA:
            if os.path.exists(folderPath):
                shutil.rmtree(folderPath)

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            os.mkdir(self.EnvSettings.getOutputFolder_depth(folderPath))
            os.mkdir(self.EnvSettings.getOutputFolder_depthLog(folderPath))
            os.mkdir(self.EnvSettings.getOutputFolder_rgb(folderPath))
            os.mkdir(self.EnvSettings.getOutputFolder_seg(folderPath))
            os.mkdir(self.EnvSettings.getOutputFolder_segcities(folderPath))

    def createWalkersBlueprintLibrary(self):
        blueprints = self.blueprint_library.filter(self.EnvSettings.walkers_filter_str)
        return blueprints

    def createVehiclesBlueprintLibrary(self):
        # Filter some vehicles library
        blueprints = self.blueprint_library.filter(self.EnvSettings.vehicles_filter_str)
        #blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints

    def __init__(self, host, port):
        self.s_weather_presets = self.find_weather_presets()
        self.s_players_actor_list= [] # The list of all actors currently spawned for main player (his car, sensor cameras ,etc)
        self.s_vehicles_list = [] # The list of all vehicle
        self.s_all_pedestrian_ids = []  # controller,walker pairs
        self.all_pedestrian_actors = [] # controller, walker pairs
        self.sensors = [] # THe list of sensors (part of s_players_actor_list
        self.map = None
        self.world = None
        self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
        self.client = None
        self.traffic_manager = None

        pygame.init()

        try:
            self.display = pygame.display.set_mode((self.EnvSettings.image_size[0], self.EnvSettings.image_size[1]),
                                                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = RenderUtils.get_font()
            self.clock = pygame.time.Clock()

            # Make paths on disk for output data
            """
            for episode_i in range(self.EnvSettings.number_of_spawnEpisodes):
                output_folder = self.EnvSettings.OUTPUT_DATA_PREFIX + str(episode_i)
                prepareOutputFolder(output_folder)
            """

            # Connect with the server
            self.client = carla.Client(host, port)
            self.client.set_timeout(self.EnvSettings.TIMEOUT_VALUE)
            self.availableMaps = self.client.get_available_maps()
            logging.log(logging.INFO, ("Available maps are: {0}").format(self.availableMaps))
            self.orig_settings = self.client.get_world().get_settings()

            for mapName in self.EnvSettings.MapsToTest:
                found = False
                for mapAvailable in self.availableMaps:
                    if mapName in mapAvailable:
                        found = True
                        break
                assert found, ("Requested test map %s is not available on the server" % mapName)

            print(dir(self.client))
            logging.log(logging.INFO, "Carla is connected")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            tb = traceback.format_exc()
            print(tb)
            self.releaseServerConnection()
            pygame.quit()

    def loadWorld(self):
        self.world = self.client.reload_world()
        # Set settings for this episode and reload the world
        settings = carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.EnvSettings.fixedFPS)
        # settings.randomize_seeds()
        self.world.apply_settings(settings)
        self.map = self.world.get_map()

        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # These are the spawnpoints for the vehicles in the map
        self.vehicles_spawn_points = self.map.get_spawn_points()

        # These are the spawnpoints for the player vehicle
        # Get one for each episode indeed, sorted by importance
        # We try to spawn the player close and with view to crosswalks
        self.spawn_points_nearcrosswalks = self.map.get_spawn_points_nearcrosswalks()
        self.player_spawn_pointsAndIndices = self.uniformSampleSpawnPoints(self.spawn_points_nearcrosswalks,
                                                                           self.EnvSettings.maxNumberOfEpisodes)

        assert len(self.player_spawn_pointsAndIndices) > 0, "There are no interesting spawn points on this map. Remove map or lower requirements from the server side"

        #self.numEpisodesToSimulate = min(len(self.player_spawn_pointsAndIndices), self.EnvSettings.maxNumberOfEpisodes)
        #logging.log(logging.INFO, 'I will simulate %d episodes' % self.numEpisodesToSimulate)
        logging.log(logging.INFO, "There are %d interesting spawn points on the map" % len(self.player_spawn_pointsAndIndices))

    def spawnEnvironment(self, NumberOfVehicles, NumberOfPedestrians, playerSpawnTransform):
        logging.log(logging.INFO, "Starting to create the environment...")

        # Spawn the player's vehicle at the given location
        playerSpawnLocation = playerSpawnTransform.location
        self.currWaypoint = self.map.get_waypoint(playerSpawnTransform.location) # This is its first waypoint

        vehiclesLib = self.blueprint_library.filter('vehicle.audi.a*')
        self.playerVehicle = self.world.spawn_actor(random.choice(vehiclesLib), playerSpawnTransform)
        self.s_players_actor_list.append(self.playerVehicle)
        self.playerVehicle.set_simulate_physics(False)

        # Spawn the camera sensors
        #------------------------------------------------------
        logging.log(logging.INFO, 'Spawning sensors...')
        # Create sensors blueprints
        cameraRgbBlueprint = self.blueprint_library.find('sensor.camera.rgb')
        cameraDepthBlueprint = self.blueprint_library.find('sensor.camera.depth')
        cameraSegBlueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        cameraBlueprints = [cameraRgbBlueprint, cameraDepthBlueprint, cameraSegBlueprint]
        for cam in cameraBlueprints:
            self.EnvSettings.configureCameraBlueprint(cam)

        # Spawn the actors
        camera_rgb = self.world.spawn_actor(cameraRgbBlueprint, self.EnvSettings.camera_front_transform, attach_to=self.playerVehicle)
        self.s_players_actor_list.append(camera_rgb)
        camera_depth = self.world.spawn_actor(cameraDepthBlueprint, self.EnvSettings.camera_front_transform, attach_to=self.playerVehicle)
        self.s_players_actor_list.append(camera_depth)
        camera_semseg = self.world.spawn_actor(cameraSegBlueprint, self.EnvSettings.camera_front_transform, attach_to=self.playerVehicle)
        self.s_players_actor_list.append(camera_semseg)
        self.sensors = {'rgb' : camera_rgb, 'depth' : camera_depth, 'seg' : camera_semseg}
        #------------------------------------------------------

        SpawnActorFunctor = carla.command.SpawnActor

        # some settings
        percentagePedestriansRunning = 0.3  # how many pedestrians will run
        percentagePedestriansCrossing = 0.6  # how many pedestrians will walk through the road

        time.sleep(SYNC_TIME)
        self.world.tick() # Be sure that player's vehicle is spawned

        blueprints_walkers  = self.createWalkersBlueprintLibrary()
        blueprints_vehicles = self.createVehiclesBlueprintLibrary()

        # --------------
        # Spawn vehicles
        # --------------
        logging.log(logging.INFO, 'Spawning vehicles')
        self.vehicles_spawn_points = sorted(self.vehicles_spawn_points, key = lambda transform : compute_distance(transform.location, playerSpawnTransform.location))
        for n, transform in enumerate(self.vehicles_spawn_points):
            if n >= NumberOfVehicles:
                break
            blueprint = random.choice(blueprints_vehicles)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.s_vehicles_list.append(vehicle)

        spawnAndDestinationPoints = []

        # -------------
        # Spawn Walkers
        # -------------
        playerSpawnForward = playerSpawnTransform.rotation.get_forward_vector()
        logging.log(logging.INFO, 'Spawning walkers...')
        walkers_list = []
        # 1. take all the random locations to spawn
        spawnAndDestinationPoints_extended = []
        # To promote having agents around the player spawn position, we randomly select F * numPedestrians locations,
        # on the navmesh, then select the closest ones to the spawn position
        numSpawnPointsToGenerate = self.EnvSettings.PedestriansSpawnPointsFactor * NumberOfPedestrians
        for i in range(numSpawnPointsToGenerate):
            loc1 = self.world.get_random_location_from_navigation()
            loc2 = self.world.get_random_location_from_navigation()
            if (loc1 != None and loc2 != None):

                # Check if one of them are in front of the car
                forward_loc1 = loc1 - playerSpawnLocation
                forward_loc2 = loc2 - playerSpawnLocation
                isLoc1InFront = dot3D(forward_loc1, playerSpawnForward)
                isLoc2InFront = dot3D(forward_loc2, playerSpawnForward)
                if isLoc1InFront or isLoc2InFront:
                    # Swap spawn with destination maybe position
                    #if isLoc1InFront == False:
                    #    loc2, loc1 = loc1, loc2
                    spawn_point = carla.Transform()
                    spawn_point.location = loc1
                    destination_point = carla.Transform()
                    destination_point.location = loc2
                    spawnAndDestinationPoints_extended.append((spawn_point, destination_point))

        # Sort the points depending on their distance to playerSpawnTransform
        spawnAndDestinationPoints_extended = sorted(spawnAndDestinationPoints_extended, key = lambda SpawnAndDestTransform : compute_distance(SpawnAndDestTransform[0].location, playerSpawnTransform.location))

        if len(spawnAndDestinationPoints_extended) > 0:
            # Now select points that are Xm depart from each other
            spawnAndDestinationPoints = [spawnAndDestinationPoints_extended[0]]
            unselected_points = []
            for pIndex in range(1, len(spawnAndDestinationPoints_extended)):
                potential_point = spawnAndDestinationPoints_extended[pIndex]
                shortedDistToAnySelected = math.inf
                for selectedPoint, destPoint in spawnAndDestinationPoints:
                    distToThisSelPoint = compute_distance(potential_point[0].location, selectedPoint.location)
                    if distToThisSelPoint < shortedDistToAnySelected:
                        shortedDistToAnySelected = distToThisSelPoint

                if shortedDistToAnySelected > self.EnvSettings.PedestriansDistanceBetweenSpawnpoints:
                    spawnAndDestinationPoints.append(potential_point)
                else:
                    unselected_points.append(potential_point)

                # Selecting enough, so leaving
                if len(spawnAndDestinationPoints) >= NumberOfPedestrians:
                    break

            # Didn't complete the list with the filter above ? just chose some random points
            diffNeeded = NumberOfPedestrians - len(spawnAndDestinationPoints)
            if diffNeeded > 0:
                U = list(np.random.choice(unselected_points, size=diffNeeded, replace=False))
                spawnAndDestinationPoints.extend(U)
            spawnAndDestinationPoints = spawnAndDestinationPoints[:NumberOfPedestrians]

            # Destination points are from the same set, but we shuffle them
            #destination_points = spawnAndDestinationPoints
            #random.shuffle(destination_points)


        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        target_points = []
        for spawn_point, target_point in spawnAndDestinationPoints:
            walker_bp = random.choice(blueprints_walkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                maxRunningSpeed = float(walker_bp.get_attribute('speed').recommended_values[2])
                maxWalkingSpeed = float(walker_bp.get_attribute('speed').recommended_values[1])
                minRunningSpeed = maxWalkingSpeed
                minWalkingSpeed = max(1.2, maxWalkingSpeed * 0.5)

                outSpeed = maxWalkingSpeed
                if random.random() > percentagePedestriansRunning:
                    # walking
                    outSpeed = minWalkingSpeed + np.random.rand() * (maxWalkingSpeed - minWalkingSpeed)
                else:
                    # running
                    outSpeed = minRunningSpeed + np.random.rand() * (maxRunningSpeed - minRunningSpeed)

                walker_speed.append(outSpeed)
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            target_points.append(target_point)
            batch.append(SpawnActorFunctor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)

        # Store from walker speeds and target points only those that succeeded
        walker_speed2 = []
        target_points2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
                target_points2.append(target_points[i])
        walker_speed = walker_speed2
        target_point = target_points2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActorFunctor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            self.s_all_pedestrian_ids.append(walkers_list[i]["con"])
            self.s_all_pedestrian_ids.append(walkers_list[i]["id"])
        all_pedestrian_actors = self.world.get_actors(self.s_all_pedestrian_ids)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.s_all_pedestrian_ids), 2):
            # start walker
            all_pedestrian_actors[i].start()
            # set walk to random point
            all_pedestrian_actors[i].go_to_location(target_points[int(i/2)].location)
            # max speed
            all_pedestrian_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        logging.log(logging.INFO, 'Spawned %d vehicles and %d walkers', len(self.s_vehicles_list), len(walkers_list))

        # Wait to have all things spawned on server side
        time.sleep(SYNC_TIME)
        self.world.tick()

        # Set auto pilot for vehicles spawned
        for v in self.s_vehicles_list:
            v.set_autopilot(True)

        logging.log(logging.INFO, 'Setting some random weather and traffic management...')
        # Now set the weather
        weather_id = np.random.choice(len(self.s_weather_presets))
        preset = self.s_weather_presets[weather_id]
        self.world.set_weather(preset[0])

        # Set the traffic management stuff
        # NOTE: the issue in the past with traffic manager was that cars were not moving after the second episode
        # To that end why i did was to:
        # - increase the timeout value to 10s and check the outputs from TM
        # - destroy the client each time between episodes (i.e. having a script that handles data gathering and
        # connects each time with a new client.)         
        self.traffic_manager = self.client.get_trafficmanager(self.EnvSettings.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(-20.0)

        time.sleep(SYNC_TIME)
        self.world.tick()

    def releaseServerConnection(self):
        # Deactivate sync mode
        if self.world == None:
            return

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def destroy_current_environment(self, client):
        if len(self.s_vehicles_list) == 0 and len(self.s_players_actor_list) == 0 and len(self.s_all_pedestrian_ids) == 0:
            logging.log(logging.INFO, 'Environment already distroyed')
            return

        logging.log(logging.INFO, 'Destroying %d vehicles' % len(self.s_vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_vehicles_list])
        self.s_vehicles_list = []

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        logging.log(logging.INFO,"Stopping the walker controllers")
        for i in range(0, len(self.all_pedestrian_actors), 2):
            self.all_pedestrian_actors[i].stop()

        logging.log(logging.INFO, f'Destroying all {len(self.s_all_pedestrian_ids)/2} walkers actors spawned')
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_all_pedestrian_ids])
        self.s_all_pedestrian_ids = []

        logging.log(logging.INFO, f"Destroying all {len(self.s_players_actor_list)/2} player\'s actors")
        client.apply_batch([carla.command.DestroyActor(x) for x in self.s_players_actor_list])
        self.s_players_actor_list = []

        time.sleep(SYNC_TIME_PLUS)
        self.world.tick()

        logging.log(logging.INFO, 'Destroying the environment')
        self.releaseServerConnection()

        logging.log(logging.INFO, "===End destroying the environment...")

    def collectSingleEpisodeData(self, outputFolder):
        logging.log(logging.INFO, "Collecting data for this episode..")
        numSimFrame = self.EnvSettings.number_of_frames_to_capture * self.EnvSettings.frame_step

        # Create the sensor management data
        dataManager = SensorsDataManagement(self.world, self.EnvSettings.fixedFPS, self.sensors)

        # Cache the folders for storing the outputs
        outputFolder_depth  = self.EnvSettings.getOutputFolder_depth(outputFolder)
        outputFolder_depthLog = self.EnvSettings.getOutputFolder_depthLog(outputFolder)
        outputFolder_rgb    = self.EnvSettings.getOutputFolder_rgb(outputFolder)
        outputFolder_seg    = self.EnvSettings.getOutputFolder_seg(outputFolder)
        output_segcities    = self.EnvSettings.getOutputFolder_segcities(outputFolder)

        tenthNumFrames = (numSimFrame / 10)
        for frameId in range(numSimFrame):
            # Process the input
            inputEv = RenderUtils.get_input_event()
            if inputEv == RenderUtils.EventType.EV_QUIT:
                return False
            elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
                self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_DEPTH
            elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
                self.RenderAsDepth = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG

            if frameId % tenthNumFrames == 0:
                print(f"{(frameId*10.0)/tenthNumFrames}%...")

            # Tick the pygame clock and world
            self.clock.tick()
            worldFrame = self.world.tick()

            # Advance the simulation and wait for the data.
            #logging.log(logging.INFO, f"Getting data for frame {worldFrame}")
            syncData = dataManager.tick(targetFrame=worldFrame, timeout=self.EnvSettings.TIMEOUT_VALUE * 100.0) # Because sometimes you forget to put the focus on server and BOOM
            #logging.log(logging.INFO, f"Data retrieved for frame {worldFrame}")

            # Take the date from world
            worldSnapshot = syncData['worldSnapshot']
            image_seg = syncData["seg"]
            image_rgb = syncData["rgb"]
            image_depth = syncData["depth"]

            # Save output to disk on each self.EnvSettings.frame_step step
            if self.EnvSettings.ENABLED_SAVING and (frameId % self.EnvSettings.frame_step == 0):
                fileName = ("%06d.png" % frameId)
                # Save RGB
                image_rgb.save_to_disk(os.path.join(outputFolder_rgb, fileName))

                # Save seg
                image_seg.save_to_disk(os.path.join(outputFolder_seg, fileName))
                image_seg.convert(carla.ColorConverter.CityScapesPalette)
                image_seg.save_to_disk(os.path.join(output_segcities, fileName))

                # Save depth
                image_depth.save_to_disk(os.path.join(outputFolder_depth, fileName))
                image_depth.convert(carla.ColorConverter.Depth)
                image_depth.save_to_disk(os.path.join(outputFolder_depthLog, fileName))
            else:
                image_seg.convert(carla.ColorConverter.CityScapesPalette)
                image_depth.convert(carla.ColorConverter.Depth)

            # Choose the next waypoint and update the car location.
            if not self.EnvSettings.STATIC_CAR:
                self.currWaypoint = random.choice( self.currWaypoint.next(1.5))
                self.playerVehicle.set_transform( self.currWaypoint.transform)
            else:
                # Apply brake
                self.playerVehicle.apply_control(carla.VehicleControl(hand_brake=True))

            # Draw the display and stats
            if self.RenderAsDepth ==  RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
                RenderUtils.draw_image(self.display, image_depth, blend=True)
            elif self.RenderAsDepth == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
                RenderUtils.draw_image(self.display, image_rgb)
                RenderUtils.draw_image(self.display, image_seg, blend=True)

            self.display.blit(self.font.render('%Press D - Depth or S - Segmentation + RGB', True, (255, 255, 255)), (8, 5))
            self.display.blit(self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)), (8, 20))

            fps = round(1.0 / worldSnapshot.timestamp.delta_seconds)
            self.display.blit(self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),(8, 38))
            pygame.display.flip()

        return True

    def collectData(self):
        if (self.client is None):
            logging.log(logging.INFO, "Can't collect data because client is not connected to server")
            return
        # Run for a number of predefined episodes
        try:
            for mapName in self.EnvSettings.MapsToTest:

                # Load the map
                self.world = self.client.load_world(mapName)

                # Do the episodes. We cycle through the spawn points if not enough
                for episodeIndex in range(self.EnvSettings.maxNumberOfEpisodes):
                    self.loadWorld()
                    # Set the current output folder
                    logging.log(logging.INFO, "Preparing episode %s\n=================", episodeIndex)
                    logging.log(logging.INFO, "Setting parameters and world %s\n=================", episodeIndex)

                    spawnPointIter = episodeIndex % len(self.player_spawn_pointsAndIndices)
                    playerSpawnTransform = self.player_spawn_pointsAndIndices[spawnPointIter][1] # The location where to spawn
                    playerSpawnIndex = self.player_spawn_pointsAndIndices[spawnPointIter][0] # The index from the original set of spawn points where to spawn

                    logging.log(logging.INFO, ('Spawning player vehicle at index %d and position (%f, %f, %f)') % (playerSpawnIndex,
                    playerSpawnTransform.location.x, playerSpawnTransform.location.y, playerSpawnTransform.location.z))


                    self.spawnEnvironment(NumberOfVehicles=self.EnvSettings.NumVehicles,
                                          NumberOfPedestrians=self.EnvSettings.NumPedestrians,
                                          playerSpawnTransform=playerSpawnTransform)

                    output_folder = self.EnvSettings.OUTPUT_DATA_PREFIX % (mapName, episodeIndex, playerSpawnIndex)
                    self.prepareOutputFolder(output_folder)
                    shouldContinue = self.collectSingleEpisodeData(output_folder)
                    self.destroy_current_environment(self.client)

                    if shouldContinue == False:
                        break
        except:
            print("Unexpected error:", sys.exc_info()[0])
            tb = traceback.format_exc()
            print(tb)
            raise

        finally:
            print("Finished, so I'm destroying the environment and reaply settings")
            self.destroy_current_environment(self.client)
            pygame.quit()

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    dc = DataCollector(host=args.host, port=args.port)
    dc.collectData()
    print('\nDone!')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nClient stoped by user.')
