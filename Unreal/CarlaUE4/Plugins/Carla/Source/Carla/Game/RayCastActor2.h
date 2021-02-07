// Fill out your copyright notice in the Description page of Project Settings.

// This code will be used to perform raycasting on a Carla map


///////// THE OLD IDEA
// The output will be a binary file containing:
// - world min, max bounding box
// - a tensor data of world size bounding box scaled by m_voxelSizeInCm
// - so if the world has 10m x 10m on X,Y axis and voxel size is 20cm then the output will a grid of 50 x 50 x height
// - the height will always be clamped to m_maxHeightInVoxels * voxel size, regardless of the world size..just as an optimization for building this output
// - For each of the element in this tensor we store:
// - RGB value + label ! So 4 bytes on each ??????

// Notes:
// - the height positions on Z axis (up) must be offset by the average height of road or sidewalks positions in the ray casting process
//   Taken from python code :
//                 if isCloudPointRoadOrSidewalk(pos, label):  # and (point_new_coord[1] < (1 * scale)):
//                  middle_height.append(point_float_coord[2])
//////////// NEW IDEA
// A combined ply file containing:
//Header:
/*  ply
    format ascii 1.0
    element vertex 120325 (N)
    property float32 x
    property float32 y
    property float32 z
    property uchar diffuse_red
    property uchar diffuse_green
    property uchar diffuse_blue
    property uchar label
    end_header
Then N items of type  3d pos, rgb of label , label code in cityscapes
    202.00 90.00 46.00 70 70 70 11

Below is the map coding From Carla to cityscapes: E.g. Wall is 11 in carla but 12 in Cityscapes and its color should be (102,102,156) in the output file
# 0 	None		Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 1 	Buildings	Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
# 2 	Fences		Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
# 3 	Other		Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 4 	Pedestrians	Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
# 5 	Poles		Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
# 6 	RoadLines	Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 7 	Roads		Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 8 	Sidewalks	Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
# 9 	Vegetation	Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
# 10 	Vehicles	Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# 11 	Walls		Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
# 12 	TrafficSigns	Label(  'traffic sign'     , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
# 13    Sky         Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
# 14    Ground      Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
# 15    Bridge      Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
# 16    RailTrack   Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
# 17    GuardRail   Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
# 18    TrafficLight    Label(  'traffic light'    , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
# 19    Static      Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 20    Dynamic     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
# 21    Water       Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 22    Terrain     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),

*/

#pragma GCC push_options
#pragma GCC optimize ("O0")



// Copy pasted from ObjectLabels.h in the Carla source code
enum class CityObjectLabel : uint8_t {
    None         =   0u,
    Buildings    =   1u,
    Fences       =   2u,
    Other        =   3u,
    Pedestrians  =   4u,
    Poles        =   5u,
    RoadLines    =   6u,
    Roads        =   7u,
    Sidewalks    =   8u,
    Vegetation   =   9u,
    Vehicles     =  10u,
    Walls        =  11u,
    TrafficSigns =  12u,
    Sky          =  13u,
    Ground       =  14u,
    Bridge       =  15u,
    RailTrack    =  16u,
    GuardRail    =  17u,
    TrafficLight =  18u,
    Static       =  19u,
    Dynamic      =  20u,
    Water        =  21u,
    Terrain      =  22u,
};

# define SPECIAL_LABEL_CROSSWALKS 34 // This doesn't exist in the citiscapes !
# define SPECIAL_LABEL_SIDEWALKS_EXTRA 35 // Extra sidewalks such as grass


#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include <map>
#include <vector>
#include <tuple>
#include "Async/AsyncWork.h"
#include "RayCastActor2.generated.h"


class RayCastTask;

UCLASS()
class CARLA_API ARayCastActor2 : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ARayCastActor2();

    UPROPERTY(VisibleAnywhere, Category = "Switch Components")
    class UPointLightComponent* PointLight1;

    UFUNCTION(BlueprintCallable, Category="RayCastPointCloud")
    virtual void PerformRaycast(const bool synchronous = false);

    virtual void PerformRaycastInternal();


    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    bool m_useLocationAsCenterPos = true;

    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    FVector m_centerPos = FVector(0.0, 0.0f, 0.0f); // From this position we take maxVoxelsDimX to right, +maxVoxelsDimY/2 in both positive and negative axes

    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    float m_voxelSizeInCm = 20;

    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    float m_maxVoxelsDimX = 256;

    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    float m_maxVoxelsDimY = 128;

    UPROPERTY(EditAnywhere, Category = "RayCastingProps")
    float m_maxVoxelsDimZ = 32;

    UPROPERTY(EditAnywhere, Category = "PathToOutput")
    FString m_pathToOutput = TEXT("/home/ciprian/Desktop/DatasetCustom/Scene1_ep0");

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
    void OptimizePointCloud();

    virtual void FindLevelBBox();
    virtual void FillMappings();
    virtual void WriteOutputPlyFiles();
    void DoSetup();
    FBox m_worldBBox;

    // For a given voxel Height, decide what is the next step to put a voxel in output file on height
    int GetVoxelResolutionSkipStep(int voxelHeight);

    // A single point cloud item
    struct PointCloudResult
    {
        FColor rgbColor;
        uint8 segLabel;
    };

    std::map<int, int> m_carlaToCitiscapesLabel;
    std::vector<std::tuple<int,int,int>> m_cityscapes_segColorByLabel;

    // Mapping from voxel coord (x,y,z) to PointCloud result
    std::map<std::tuple<int, int, int>, PointCloudResult> m_lastRaycastRes;

    // Last street level minimum detected by the raycasting
    int m_lastRayCast_streetLevelMin;

    // From a world position given in centimeters, get the output X,y,Z coord in voxels space
    void getVoxelCoordFromWorldCoord(const FVector& worldPosInCM, int &outX, int &outY, int &outZ);
    FVector getWorldCoordFromVoxelCoord(const FVector& voxelsPos);

    virtual bool ShouldTickIfViewportsOnly() const override { return true;}

    bool m_isSetupDone = false;

    FAsyncTask<RayCastTask>* m_currentRaycastTask;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	int getCarlaToCityscapesLabel(int carlaLabel);
	std::tuple<int, int, int> getCityscapesSegColorOfLabel(int cityscapesLabel);
};

/*PrimeCalculateAsyncTask is the name of our task
FNonAbandonableTask is the name of the class I've located from the source code of the engine*/
class RayCastTask : public FNonAbandonableTask {
    ARayCastActor2 *m_raycastActorInst = nullptr;

public:
    /*Default constructor*/
    RayCastTask(ARayCastActor2 *objInstance) {
        m_raycastActorInst = objInstance;
    }

    /*This function is needed from the API of the engine.
    My guess is that it provides necessary information
    about the thread that we occupy and the progress of our task*/
    FORCEINLINE TStatId GetStatId() const {
        RETURN_QUICK_DECLARE_CYCLE_STAT(RayCastTask, STATGROUP_ThreadPoolAsyncTasks);
    }

    /*This function is executed when we tell our task to execute*/
    void DoWork() {
        m_raycastActorInst->PerformRaycastInternal();
        GLog->Log("End of the raycast operation");
    }
};

#pragma GCC pop_options
