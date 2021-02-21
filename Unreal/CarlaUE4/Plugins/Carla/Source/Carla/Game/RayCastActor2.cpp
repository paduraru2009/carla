// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB). This work is licensed under the terms of the MIT license. For a copy, see <https://opensource.org/licenses/MIT>.


#include "RayCastActor2.h"
#include "DrawDebugHelpers.h"
#include "CollisionQueryParams.h"
#include "Engine/World.h"
#include "Engine/StaticMeshActor.h"
#include "EngineUtils.h"
#include "Misc/AssertionMacros.h"
#include "GenericPlatform/GenericPlatform.h"
#include <fstream>
#include "Math/NumericLimits.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include <map>
#include <algorithm>
#include <vector>
#include <assert.h>

#pragma GCC push_options
#pragma GCC optimize ("O0")

bool g_restartDebug = false;

static bool rayCastEnable = true;

void convertFromCMToMeters(float& x, float &y, float &z)
{
    float CM_TO_M = 0.01f;
    x *= CM_TO_M;
    y *= CM_TO_M;
    z *= CM_TO_M;
}


#define APPLY_HIT_LIMIT_HACK // Check the comment where it is being used to understand the reason

// Sets default values
ARayCastActor2::ARayCastActor2()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
    m_lastRayCast_streetLevelMin = TNumericLimits<int>::Max();
    bAllowTickBeforeBeginPlay = true;
}

// Called when the game starts or when spawned
void ARayCastActor2::BeginPlay()
{
	AActor::BeginPlay();

	DoSetup();

    //g_restartDebug = true;
}

void ARayCastActor2::DoSetup()
{
    if (m_isSetupDone == false)
    {
        FindLevelBBox();
        FillMappings();
        //check (m_worldBBox.Min.X > -10000);

        m_isSetupDone = true;
    }
}

bool isCarlaLabelForStreetLevelObject(const uint8_t carlaLabel)
{
    return carlaLabel == (uint8_t)CityObjectLabel::RoadLines ||
            carlaLabel == (uint8_t)CityObjectLabel::Roads ||
            carlaLabel == (uint8_t)CityObjectLabel::Sidewalks ||
            carlaLabel == (uint8_t) CityObjectLabel::Ground ||
            carlaLabel == (uint8_t) CityObjectLabel::Terrain ||
            carlaLabel == SPECIAL_LABEL_SIDEWALKS_EXTRA ||
            carlaLabel == SPECIAL_LABEL_CROSSWALKS;
}

bool isCarlaLabelForTerrainOrWater(const uint8_t carlaLabel)
{
    return carlaLabel == (uint8_t)CityObjectLabel::Terrain || carlaLabel == (uint8_t)CityObjectLabel::Water;
}

// The purpose is to fill intermediate voxels on the Z axis in a compressed way for the output not at each mini voxel
int ARayCastActor2::GetVoxelResolutionSkipStep(int voxelHeight)
{
    // This is the resolution, we want a voxel size before 1m at 20 cm then aove 1 m we consider a voxel as 100
    static constexpr int TARGET_VOXEL_SIZE_1M = 20;
    static constexpr int TARGET_VOXEL_SIZE_ABOVE_1M = 100;

    const int currVoxelSizeRounded = (int)m_voxelSizeInCm;
    check(TARGET_VOXEL_SIZE_1M % currVoxelSizeRounded == 0);
    check(TARGET_VOXEL_SIZE_ABOVE_1M % currVoxelSizeRounded == 0);

    FVector voxelHeightPos = getWorldCoordFromVoxelCoord(FVector(0, 0, voxelHeight));
    convertFromCMToMeters(voxelHeightPos.X, voxelHeightPos.Y, voxelHeightPos.Z);
    if (voxelHeightPos.Z < 1.01f) // Less than 1 m ?
    {
        return TARGET_VOXEL_SIZE_1M / currVoxelSizeRounded;
    }
    else
    {
        return TARGET_VOXEL_SIZE_ABOVE_1M / currVoxelSizeRounded;
    }
}

void ARayCastActor2::FindLevelBBox() //__attribute__ ((optnone))
{
    //TArray<AStaticMeshActor*> staticMeshActors;
    m_worldBBox.IsValid = 1;

    for (TActorIterator<AStaticMeshActor> ActorItr(GetWorld()); ActorItr; ++ActorItr)
    {
        //staticMeshActors.Add(*ActorItr);

        FVector origin, boxExtent;
        ActorItr->GetActorBounds(true, origin, boxExtent);
        FVector actorMin = origin - boxExtent;
        FVector actorMax = origin + boxExtent;

        m_worldBBox += actorMin;
        m_worldBBox += actorMax;
    }
}

void ARayCastActor2::getVoxelCoordFromWorldCoord(const FVector& worldPosInCM, int &outX, int &outY, int &outZ)
{
    outX = worldPosInCM.X / m_voxelSizeInCm;
    outY = worldPosInCM.Y / m_voxelSizeInCm;
    outZ = worldPosInCM.Z / m_voxelSizeInCm;
}

FVector ARayCastActor2::getWorldCoordFromVoxelCoord(const FVector& voxelsPos)
{
    FVector out = FVector(voxelsPos.X * m_voxelSizeInCm, voxelsPos.Y * m_voxelSizeInCm, voxelsPos.Z * m_voxelSizeInCm);
    return out;
}

void ARayCastActor2::WriteOutputPlyFiles()
{
#if 1
    const FString segPath = FString::Printf(TEXT("%s/00000_seg.ply"), *m_pathToOutput);
    std::string segPathStr(TCHAR_TO_UTF8(*segPath));
    const char* outputFileNameSeg = segPathStr.c_str();

    const FString segColorPath = FString::Printf(TEXT("%s/00000_segColor.ply"), *m_pathToOutput);
    std::string segColorPathStr(TCHAR_TO_UTF8(*segColorPath));
    const char* outputFileNameSegColor = segColorPathStr.c_str();

    const FString rgbPath = FString::Printf(TEXT("%s/00000.ply"), *m_pathToOutput);
    std::string rgbPathStr(TCHAR_TO_UTF8(*rgbPath));
    const char* outputFileNameRGB = rgbPathStr.c_str();

    const FString transPath = FString::Printf(TEXT("%s/translation.txt"), *m_pathToOutput);
    std::string transPathStr(TCHAR_TO_UTF8(*transPath));
    const char* outputTranslationFileName = transPathStr.c_str();
#else
    const char* outputFileNameSeg = "/home/ciprian/Desktop/DatasetCustom/Scene1/00000_seg.ply";
    const char* outputFileNameSegColor = "/home/ciprian/Desktop/DatasetCustom/Scene1/00000_segColor.ply";
    const char* outputFileNameRGB = "/home/ciprian/Desktop/DatasetCustom/Scene1/00000.ply";
    const char* outputTranslationFileName = "/home/ciprian/Desktop/DatasetCustom/Scene1/translation.txt";
#endif

    std::ofstream outputFile_seg, outputFile_rgb, outputFile_segColor, outputFile_translation;
    outputFile_seg.open(outputFileNameSeg);
    ensureMsgf(outputFile_seg.is_open(), TEXT("Couldn't open file for writing!!!!!!"));
    check(outputFile_seg.is_open());
    outputFile_rgb.open(outputFileNameRGB);
    outputFile_segColor.open(outputFileNameSegColor);
    outputFile_translation.open(outputTranslationFileName);

    UE_LOG(LogTemp, Error, TEXT("!!! Will write OUTPUT seg to %s" ), *FString(outputFileNameSeg));

    FVector referencePos2D = m_useLocationAsCenterPos ? GetActorLocation() : m_centerPos;
    referencePos2D.Z = 0.0f;

#if 0
    // Write headers first
    std::ofstream *ar[2] = {&outputFile_seg, &outputFile_rgb};

    for (int i = 0; i < 2; i++)
    {
        std::ofstream& outputFile = *ar[i];
        outputFile << "ply" << std::endl;
        outputFile << "format ascii 1.0" << std::endl;
        outputFile << "element vertex " << m_lastRaycastRes.size() << std::endl;
        outputFile << "property float32 x" << std::endl;
        outputFile << "property float32 y" << std::endl;
        outputFile << "property float32 z" << std::endl;
        outputFile << "property uchar diffuse_red" << std::endl;
        outputFile << "property uchar diffuse_green" << std::endl;
        outputFile << "property uchar diffuse_blue" << std::endl;
        outputFile << "property uchar label" << std::endl;
        outputFile << "end_header" << std::endl;
    }
#else
    // Write the header for the segmentation file
    outputFile_seg << "ply" << std::endl;
    outputFile_seg << "format ascii 1.0" << std::endl;
    outputFile_seg << "element vertex " << m_lastRaycastRes.size() << std::endl;
    outputFile_seg << "property float32 x" << std::endl;
    outputFile_seg << "property float32 y" << std::endl;
    outputFile_seg << "property float32 z" << std::endl;
    outputFile_seg << "property uchar label" << std::endl;
    outputFile_seg << "end_header" << std::endl;

    // Write the header for the seg rgb file
    outputFile_segColor << "ply" << std::endl;
    outputFile_segColor << "format ascii 1.0" << std::endl;
    outputFile_segColor << "element vertex " << m_lastRaycastRes.size() << std::endl;
    outputFile_segColor << "property float32 x" << std::endl;
    outputFile_segColor << "property float32 y" << std::endl;
    outputFile_segColor << "property float32 z" << std::endl;
    outputFile_segColor << "property uchar diffuse_red" << std::endl;
    outputFile_segColor << "property uchar diffuse_green" << std::endl;
    outputFile_segColor << "property uchar diffuse_blue" << std::endl;
    outputFile_segColor << "end_header" << std::endl;


    // Write the header for the rgb file
    outputFile_rgb << "ply" << std::endl;
    outputFile_rgb << "format ascii 1.0" << std::endl;
    outputFile_rgb << "element vertex " << m_lastRaycastRes.size() << std::endl;
    outputFile_rgb << "property float32 x" << std::endl;
    outputFile_rgb << "property float32 y" << std::endl;
    outputFile_rgb << "property float32 z" << std::endl;
    outputFile_rgb << "property uchar diffuse_red" << std::endl;
    outputFile_rgb << "property uchar diffuse_green" << std::endl;
    outputFile_rgb << "property uchar diffuse_blue" << std::endl;
    outputFile_rgb << "end_header" << std::endl;
#endif

    for (const auto &it : m_lastRaycastRes)
    {
        const auto& pos = it.first;
        const PointCloudResult& pcr = it.second;
        const uint8 carlaLabel = (uint8)pcr.segLabel;
        const auto label_citiscapes = m_carlaToCitiscapesLabel[carlaLabel];

        float vx = 0, vy = 0, vz = 0;
        std::tie(vx, vy, vz) = pos;

        // Probably not need to extract here
        // vz -= m_lastRayCast_streetLevelMin;
        FVector worldPos = getWorldCoordFromVoxelCoord(FVector(vx, vy, vz));
        worldPos -= referencePos2D;
        convertFromCMToMeters(worldPos.X, worldPos.Y, worldPos.Z);


        outputFile_seg << worldPos.X << " " << worldPos.Y << " " << worldPos.Z << " ";
        outputFile_rgb << worldPos.X << " " << worldPos.Y << " " << worldPos.Z << " ";
        outputFile_segColor << worldPos.X << " " << worldPos.Y << " " << worldPos.Z << " ";

        int segR = 0, segG = 0, segB = 0;
        std::tie(segR, segG, segB) = m_cityscapes_segColorByLabel[carlaLabel];

        outputFile_segColor << segR << " " << segG << " " << segB << std::endl;
        outputFile_seg << (int)carlaLabel << std::endl;
        outputFile_rgb << (int)pcr.rgbColor.R << " " << (int)pcr.rgbColor.G << " " << (int)pcr.rgbColor.B << std::endl;
    }

    FVector worldTranslationPos = referencePos2D;
    convertFromCMToMeters(worldTranslationPos.X, worldTranslationPos.Y, worldTranslationPos.Z);
    outputFile_translation << worldTranslationPos.X << " " << worldTranslationPos.Y << " " << worldTranslationPos.Z;

    outputFile_seg.close();
    outputFile_rgb.close();
    outputFile_segColor.close();
    outputFile_translation.close();
    UE_LOG(LogTemp, Error, TEXT("!!! Wroteee OUTPUT to %s" ), *FString(outputFileNameSeg));
}

int32 GetMaterialByFaceIndex(const UStaticMesh* mesh, const int32& faceIndex)
{
    const FStaticMeshLODResources& lodRes = mesh->GetLODForExport(0);
    int32 totalSection =  lodRes.Sections.Num();

    int32 totalFace = lodRes.GetNumTriangles();
    if (faceIndex > totalFace)
    {
        ///Wrong faceIndex.
        UE_LOG(LogTemp, Warning, TEXT("GetMaterialbyFaceIndex Faild, Wrong faceIndex!"));
        return -1;
    }

    int32 totalTriangleIndex = 0;
    for (int32 sectionIndex = 0; sectionIndex < totalSection; ++sectionIndex)
    {
        FStaticMeshSection currentSection = lodRes.Sections[sectionIndex];
        /*Check for each triangle, so we can know which section our faceIndex in. For Low Performance, do not use it.
        for (int32 triangleIndex = totalTriangleIndex; triangleIndex < (int32)currentSection.NumTriangles + totalTriangleIndex; ++triangleIndex)
        {

            if (faceIndex == triangleIndex)
            {
                //return sectionIndex;
                return currentSection.MaterialIndex;
            }
        }
        totalTriangleIndex += lodRes.Sections[sectionIndex].NumTriangles;
        */

        ///the triangle index is sorted by section, the larger sectionIndex contains larger triangle index, so we can easily calculate which section the faceIndex in.Performance Well.
        int32 triangleIndex = totalTriangleIndex;
        if (faceIndex >= triangleIndex && faceIndex < triangleIndex + (int32)lodRes.Sections[sectionIndex].NumTriangles)
        {
            //return sectionIndex;
            return currentSection.MaterialIndex;
        }
        totalTriangleIndex += lodRes.Sections[sectionIndex].NumTriangles;
    }

    /// did not find the face in the mesh.
    UE_LOG(LogTemp, Warning, TEXT("GetMaterialByFaceIndex, did not find it!"));
    return -2;
}

inline uint32_t hash_str_uint32(const std::string& str)
{
    uint32_t hash = 0x811c9dc5;
    uint32_t prime = 0x1000193;

    for(int i = 0; i < str.size(); ++i)
    {
        uint8_t value = str[i];
        hash = hash ^ value;
        hash *= prime;
    }

    return hash;
}

void ARayCastActor2::PerformRaycast(const bool synchronous)
{
    if (!rayCastEnable)
    {
         return;
    }


    // If already started, wait for task completion
    if (m_currentRaycastTask != nullptr)
    {
        if (!m_currentRaycastTask->IsDone()) {
            m_currentRaycastTask->EnsureCompletion();
        }
        delete m_currentRaycastTask;
        m_currentRaycastTask = nullptr;
    }

    m_currentRaycastTask = new FAsyncTask<RayCastTask>(this);
    m_currentRaycastTask->StartBackgroundTask();

    if (synchronous)
    {
        m_currentRaycastTask->EnsureCompletion();
    }
}

void ARayCastActor2::PerformRaycastInternal()  //__attribute__ ((optnone))
{
    TArray<FColor> debugColors = {FColor::Red, FColor::Black, FColor::Blue, FColor::Magenta, FColor::Orange, FColor::Yellow, FColor::Cyan};

    m_lastRaycastRes.clear();
    FVector centerWorld = m_useLocationAsCenterPos ? GetActorLocation() : m_centerPos;
    const int halfDimYVoxels = int(m_maxVoxelsDimY / 2);
    const int halfDimZVoxels = int(m_maxVoxelsDimZ / 2);

    int centerX = 0, centerY = 0, centerZ = 0;
    getVoxelCoordFromWorldCoord(centerWorld, centerX, centerY, centerZ);

    m_lastRayCast_streetLevelMin = TNumericLimits<int>::Max(); // Minimum of the street level voxels on height; normalization purposes to offset the scene at this min point such that it is considered 0
    //UE_LOG(LogTemp, Warning, TEXT("GetMaterialbyFaceIndex Faild, Wrong faceIndex!"));
    // Step 1: Iterate over each voxel in the requested grid (X,Y,Z)
    const int voxelCellsToEvaluate2D_total = (float) m_maxVoxelsDimX * m_maxVoxelsDimY;
    int voxelCellsEvaluated_total = 0;
    int voxelCellsEvaluated_slice = 0;
    const int completionLogDebugRate = std::max(voxelCellsToEvaluate2D_total/10, 1); // At each 100 new voxel give us some feedback...

    static int isDebugging = 1;
    static int debugVoxelX = -1; //1108;
    static int debugVoxelY = -1; //2290;

    const int MAX_RETRIES = 30;

    int retries = 0;

    struct ObjectAndExtents
    {
        AActor* actor;
        FVector origin;
        FVector extent;
        int label;
    };

    struct LastRayDebugInfoStep
    {
        bool isHit;
        char actorHitName[256];
        FVector hitLocation;
        const AActor* hitActor;
        const UPrimitiveComponent* hitComponent;
        FVector rayStartPos;
        FVector rayEndPos;
        int label;
        bool isSideWalkOrRoadOrCrossWalk;
        bool isTerrainOrWaterHit;
        bool isHitNext;
        char actorHitNextName[256];
        bool last_isHitAbove;
        char last_actorHitBackName[256];
        bool continueSearch;

        LastRayDebugInfoStep()
        {
            isHit = false;
            actorHitName[0] = 0;
            hitLocation = FVector(0.0f, 0.0f, 0.0f);
            hitActor = nullptr;
            hitComponent = nullptr;
            rayStartPos = rayEndPos = FVector(0.0f, 0.0f, 0.0f);
            label = -1;
            isSideWalkOrRoadOrCrossWalk = false;
            isTerrainOrWaterHit = false;
            isHitNext = false;
            actorHitNextName[0] = 0;
            last_isHitAbove = false;
            last_actorHitBackName[0] =0;
            continueSearch = false;
        }
    };

    struct LastRayDebugInfo
    {
        std::vector<LastRayDebugInfoStep> steps;
    };

    std::map<AActor*, ObjectAndExtents> allStaticObjectsMap;

    for (int voxelX = centerX; voxelX < centerX + m_maxVoxelsDimX; voxelX++)
    {
        for (int voxelY = centerY - halfDimYVoxels; voxelY < centerY + halfDimYVoxels; voxelY++)
        {
            // Debug thing
            LastRayDebugInfo debugInfo;
            if (isDebugging)
            {
                if (debugVoxelX != -1 && debugVoxelY != -1)
                {
                    voxelX = debugVoxelX;
                    voxelY = debugVoxelY;
                }
            }

            voxelCellsEvaluated_slice++;

            const int topVoxelCoord = centerZ ; //+ halfDimZVoxels;
            const FVector topWorldVec = getWorldCoordFromVoxelCoord(FVector(voxelX, voxelY, topVoxelCoord));
            const int bottomVoxelCoord = centerZ - m_maxVoxelsDimZ;

            const FVector bottomWorldVec = getWorldCoordFromVoxelCoord(FVector(voxelX, voxelY, bottomVoxelCoord));

            // For each position perform recursive raycasting and fill in the output data structure
            FHitResult *HitResult = nullptr;
            //GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("Screen Message"));

            // Take each voxel and perform raycasting..then put the results in an output map
            FVector downDir = FVector(0.0, 0.0, -1.0);
            FVector start = topWorldVec;
            FVector end = bottomWorldVec;

            FColor hitColor = FColor::Green;

            TArray<AActor *> ignoredActors;
            ignoredActors.Add(this);

            FCollisionQueryParams CollisionParams;
            CollisionParams.bTraceComplex = true;
            CollisionParams.bReturnFaceIndex = false;
            CollisionParams.bReturnPhysicalMaterial = false;
            CollisionParams.AddIgnoredActors(ignoredActors);

            const ECollisionChannel CollisionChannelTOuse = ECC_WorldStatic;

            FVector currStart = start;
            FVector prevStart = start;
            bool continueSearch = true;
            int debugColorIndex = 0;

            bool debug_wasGroundDiscovered = false;

            int prevLabel = -1;
            while (continueSearch)
            {
                continueSearch = false;

                FHitResult OutHit;
                const bool isHit = GetWorld()->LineTraceSingleByChannel(OutHit, currStart, end, CollisionChannelTOuse,
                                                                        CollisionParams);

                FVector hitLocation = end;
                hitColor = FColor::Green;

                LastRayDebugInfoStep rayDebugInfoStep;
                rayDebugInfoStep.isHit = isHit;
                const AActor* hitActor = OutHit.GetActor();
                const UPrimitiveComponent* hitComponent = OutHit.GetComponent();

                rayDebugInfoStep.hitActor = hitActor;
                rayDebugInfoStep.hitComponent = hitComponent;
                rayDebugInfoStep.rayStartPos  = currStart;

                int label = (int)CityObjectLabel::Static;
                if (isHit && hitActor && hitComponent)
                {
                    const AActor* actorHit = OutHit.GetActor();
                    const char* actorHitName = TCHAR_TO_ANSI(*actorHit->GetName());
                    strncpy(rayDebugInfoStep.actorHitName, actorHitName, 255);

                    const uint32_t hashCodeForActorName = hash_str_uint32(std::string(actorHitName));
                    const FColor debugRGBForActor = debugColors[hashCodeForActorName % debugColors.Num()];

#if 0
                    const bool isStreetFirstHit = !strstr(actorHitName, "road") && !strstr(actorHitName, "Road");
                    if (isStreetFirstHit)
                    {
                        int a = 3;
                        a++;
                    }
#endif

# if 0
                    TArray<UStaticMeshComponent*> Components;
                    actorHit->GetComponents<UStaticMeshComponent>(Components);

                    int faceindex = OutHit.FaceIndex;

                    for( int32 i=0; i<Components.Num(); i++ )
                    {
                        UStaticMeshComponent* StaticMeshComponent = Components[i];
                        UStaticMesh* StaticMesh = StaticMeshComponent->GetStaticMesh();
                        if (StaticMesh)
                        {

                            int32 materialId = GetMaterialByFaceIndex(StaticMesh, faceindex);
                            UMaterialInterface* material = StaticMesh->GetMaterial(materialId);
                            UMaterial* materialU = material->GetMaterial();
                            auto color = materialU->BaseColor;

                            //StaticMesh->GetSourceModel(0).MeshDescription.Get()-
                            TArray<FColor> outColorso;
                            StaticMesh->RenderData.Get()->LODResources[0].VertexBuffers.ColorVertexBuffer.GetVertexColors(outColorso);

                            TMap<FVector, FColor> VCMap;
                            StaticMesh->GetVertexColorData(VCMap);

                            FPositionVertexBuffer* VertexBuffer = &StaticMesh->RenderData->LODResources[0].VertexBuffers.PositionVertexBuffer;
                            FColorVertexBuffer* ColorBuffer = &StaticMesh->RenderData->LODResources[0].VertexBuffers.ColorVertexBuffer;

                            TArray<FColor> outColors;
                            ColorBuffer->GetVertexColors(outColors);
                            if (outColors.Num() > 0)
                            {
                                FColor c = outColors[0];
                                c.A = 0;
                            }

                            int  n = VCMap.Num();
                            for (int j = 0; j < n; j++)
                            {
                                //FColor c = VCMap[j].;
                                //printf("%d", c.A);
                            }
                        }
                    }
#endif

#ifdef APPLY_HIT_LIMIT_HACK
                    // If we hit a road below ground limit to ground as 0 - this is the standard in unreal carla maps we might now want to mess with this
                    // The problem is that some objects do not have a road below them, other than a support road located at -150 !
                    if (strstr(actorHitName, "Road"))
                    {
                        if (OutHit.Location.Z < 0)
                        {
                            OutHit.Location.Z = 0.0f;
                        }
                    }
#endif

                    hitLocation = OutHit.Location;

                    rayDebugInfoStep.hitLocation = hitLocation;
                    continueSearch = true;
                    hitColor = debugColors[debugColorIndex++];
                    debugColorIndex = debugColorIndex % debugColors.Num();

                    FCollisionQueryParams CollisionParamsWithoutThisHitObject = CollisionParams;
                    CollisionParams.AddIgnoredActor(actorHit); // Add this object to ignored object from now own

                    // Now try to find the boundary of this object label
                    //-------------------
                    FVector actorHitOrigin, actorHitExtent;
                    actorHit->GetActorBounds(false, actorHitOrigin, actorHitExtent);

                    // By default, put the entire bounding box of the entity
                    float actorHit_worldZStart = hitLocation.Z; //currStart.Z;
                    float actorHit_worldZEnd = FMath::Max(actorHitOrigin.Z - actorHitExtent.Z, bottomWorldVec.Z);
                    FVector nextCurrStart = currStart; // By default, the next position to start a raycast from
                    nextCurrStart.Z = actorHit_worldZEnd;

#ifdef APPLY_HIT_LIMIT_HACK
                    actorHit_worldZStart = FMath::Max(actorHit_worldZStart, 0.0f);
                    actorHit_worldZEnd = FMath::Max(actorHit_worldZEnd, 0.0f);
#endif

                    // Check along the ray by projecting a new ray below to see where it hits without this object
                    FHitResult OutHitNext;
                    const bool isHitNext = GetWorld()->LineTraceSingleByChannel(OutHitNext, hitLocation, end, CollisionChannelTOuse, CollisionParams);
                    rayDebugInfoStep.isHitNext = isHitNext;
                    if (isHitNext && OutHitNext.GetActor() != nullptr)
                    {
                        const char* actorHitNextName = TCHAR_TO_ANSI(*OutHitNext.GetActor()->GetName());
                        strncpy(rayDebugInfoStep.actorHitNextName, actorHitNextName, 255);
                        const AActor* actorHitNextActor = OutHitNext.GetActor();

                        // Put first the ending of this object at the next hit location
                        actorHit_worldZEnd = FMath::Max(actorHit_worldZEnd, OutHitNext.Location.Z);

                        // Now that we obtained a new hit without this object, project back a ray (up) to see the real boundary of the object found initially
                        FVector raycastBackStart = OutHitNext.Location;

                        static float displacementSafety = 10.0;
                        raycastBackStart.Z -= displacementSafety; // A little below the current hit to be sure that we don't hit it again
                        FVector raycastBackEnd = currStart;
                        raycastBackEnd.Z += 100.0f; // A little above to be sure that we hit the initial object

                        CollisionParamsWithoutThisHitObject.AddIgnoredActor(OutHitNext.GetActor()); // Ignore from raycast back the actor hit
                        FHitResult OutHitBack;
                        const AActor* lastHitActorByRayBack = nullptr; // Because ignore thing doesn't ignore it sometimes :(
                        bool foundPreciseZBoundary = false;
                        while (true)
                        {
                            const bool isHitAbove = GetWorld()->LineTraceSingleByChannel(OutHitBack, raycastBackStart, raycastBackEnd, CollisionChannelTOuse, CollisionParamsWithoutThisHitObject);
                            rayDebugInfoStep.last_isHitAbove = isHitAbove;

                            const AActor* actorHitBackActor = OutHitNext.GetActor();

                               if (!isHitAbove || actorHitBackActor == nullptr || actorHitBackActor == lastHitActorByRayBack)
                                break;

                            if (OutHitBack.Actor.Get() == nullptr)
                                break;

                            const char* actorHitBackName = TCHAR_TO_ANSI(*OutHitBack.Actor.Get()->GetName());
                            strncpy(rayDebugInfoStep.last_actorHitBackName, actorHitBackName, 255);
                            // We collide with something else, go above it

                            //if (actorHitBackActor != actorHit)
                            if (strcmp(actorHitBackName, actorHitName)) // Not the actor that we are looking for, continue searching above
                            {
                                raycastBackStart.Z = OutHitBack.Location.Z - displacementSafety;
                                CollisionParamsWithoutThisHitObject.AddIgnoredActor(actorHitBackActor);
                                lastHitActorByRayBack = actorHitBackActor;
                            }
                            else
                            {
                                // Limit Z to the hit location, it should be more precise now
                                actorHit_worldZEnd = FMath::Max(OutHitBack.Location.Z, actorHit_worldZEnd) - displacementSafety;
                                foundPreciseZBoundary = true;
                                break;
                            }

                        }

                        // If we found a precise Z boundary then use it. If not, then fallback to the previous starting pos.
                        // That object will be ignored from collision so it would still work.
                        // This is to fix the problem when the next object is exactly at the boundary between object and it will be missed if we use a fixed position !

#ifdef APPLY_HIT_LIMIT_HACK
                        actorHit_worldZStart = FMath::Max(actorHit_worldZStart, 0.0f);
                        actorHit_worldZEnd = FMath::Max(actorHit_worldZEnd, 0.0f);
#endif
                        nextCurrStart.Z = actorHit_worldZEnd;

                    }

#if 0
                    // Get the next hit point ignoring the current object
                    FHitResult OutHit2;
                    const bool isHit2 = GetWorld()->LineTraceSingleByChannel(OutHit2, currStart, end, ECC_Visibility, CollisionParams);
                    if (isHit2)
                    {
                        const AActor* actorHit2 = OutHit2.Actor.Get();
                        const char* actorName2 = TCHAR_TO_ANSI(*actorHit2->GetName());

                        if (strstr(actorName2, "road") || strstr(actorName2, "Road"))
                        {
                            int a = 3;
                            a++;
                        }

                        // Now if we have another hit, check the boundary of the previously hit object by performing a raycast back to the original object
                        FHitResult OutHit3;
                        FVector raycastBackEnd = OutHit.Location;
                        FVector raycastBackStart = OutHit2.Location;
                        raycastBackEnd.Z += 10.0f; // We add this to be sure that we get that object
                        raycastBackStart.Z += 10.0f;

                        bool isHit3 = GetWorld()->LineTraceSingleByChannel(OutHit3, raycastBackStart, raycastBackEnd, ECC_Visibility,
                                                                                     CollisionParamsWithoutThisHitObject);

                        // If we got a another hit and this is between the first location and second one, it means that there is a gap between them so we should update the boundings
                        if (isHit3 && (OutHit.Location.Z > OutHit3.Location.Z && OutHit3.Location.Z > OutHit2.Location.Z))
                        {
                            const AActor* actorHit3 = OutHit3.Actor.Get();
                            const char* actorName3 = TCHAR_TO_ANSI(*actorHit3->GetName());

                            actorHit_worldZEnd = OutHit3.Location.Z;
                            currStart = OutHit3.Location; // Update also the next starting position of the ray
                        }
                        else if (isStreetFirstHit)
                        {
                            int a  = 3;
                            a++;
                        }
                    }
                    else
                    {
                        int a = 3;
                        a++;
                    }
#endif
                    // Update the current start position (for next raycast)
                    currStart = nextCurrStart;
                    rayDebugInfoStep.rayEndPos = currStart;

                    // Establish the label type
                    //-----------------------------------------------------------
                    auto* componentHit = OutHit.GetComponent();
                    if (componentHit != nullptr)
                    {
                        label = (int) OutHit.GetComponent()->CustomDepthStencilValue;


                        if (label == (int)CityObjectLabel::Ground)
                        {
                            label = SPECIAL_LABEL_SIDEWALKS_EXTRA; // Could also be SIDEWALKS !!
                        }
                        else
                        {
                            // Pedestrians can walk on this !
                            char const *specialRoad_to_sidewalkCases[] = {"Road_Sidewalk", "Road_Curb"};//, "Road_Grass",};
                            const bool specialRoad_to_sidewalkIsExtra[] = {false, true,
                                                                           true}; // True if this is extra sidewalk class
                            const int numSpecialCases =
                                    sizeof(specialRoad_to_sidewalkCases) / sizeof(specialRoad_to_sidewalkCases[0]);

                            for (int roadSpecialIter = 0; roadSpecialIter < numSpecialCases; roadSpecialIter++) {
                                if (strcasestr(actorHitName, specialRoad_to_sidewalkCases[roadSpecialIter])) {
                                    label = (!specialRoad_to_sidewalkIsExtra[roadSpecialIter]
                                             ? (int) CityObjectLabel::Sidewalks : SPECIAL_LABEL_SIDEWALKS_EXTRA);
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        label = (int)CityObjectLabel::Static;
                    }

                    bool isCrossWalk = strcasestr(actorHitName, "crosswalk");
                    const bool isSideWalkOrRoadOrCrossWalk = isCarlaLabelForStreetLevelObject(label) || isCrossWalk;
                    const bool isTerrainOrWaterHit = isCarlaLabelForTerrainOrWater(label);

                    // Check if crosswalk is really the mesh below
                    // Basically if we have "Road_" below or above this then yes it must be crosswalk. If not, then
                    // the crosswalk geometry is a bit extended naively above pedestrian walking areas / terrain
                    if (isSideWalkOrRoadOrCrossWalk)
                    {
                        const char* nextHitActorName = nullptr;
                        const char *nextNextHitActorName = nullptr;

                        // Check the two next below actors. If one of them is a crosswalk it means that the point is actually on a crosswalk
                        FHitResult OutHitNextRoad;
                        const bool isHit2 = GetWorld()->LineTraceSingleByChannel(OutHitNextRoad, hitLocation, end, CollisionChannelTOuse, CollisionParams);
                        if (isHit2)
                        {
                            const AActor* nextHitActor = OutHitNextRoad.GetActor();
                            if (nextHitActor) {
                                nextHitActorName = TCHAR_TO_ANSI(*nextHitActor->GetName());
                                CollisionParams.AddIgnoredActor(nextHitActor);

                                FHitResult OutHitNextRoad2;
                                const bool isHit3 = GetWorld()->LineTraceSingleByChannel(OutHitNextRoad2, start, end,
                                                                                         CollisionChannelTOuse,
                                                                                         CollisionParams);
                                const AActor *nextHitACtor2 = OutHitNextRoad2.GetActor();

                                if (isHit3 && nextHitACtor2) {
                                    nextNextHitActorName = TCHAR_TO_ANSI(*nextHitACtor2->GetName());
                                }
                            }
                        }

                        // If already cross walk we must confirm that below is street level
                        if (isCrossWalk)
                        {
                            bool isStreetBelow = nextHitActorName &&
                                                    (strcasestr(nextHitActorName, "Road_Road") ||
                                                        strcasestr(nextHitActorName, "Road_Curb"));

                            // Maybe it is an intersection of crosswalks
                            if (!isStreetBelow && strcasestr(nextHitActorName, "crosswalk"))
                            {
                                isStreetBelow = nextNextHitActorName &&
                                                           (strcasestr(nextNextHitActorName, "Road_Road") ||
                                                            strcasestr(nextNextHitActorName, "Road_Curb"));

                            }

                            if (!isStreetBelow)
                            {

                                isCrossWalk = false;
                            }
                        }
                        else if (strcasestr(actorHitName, "Road_Road"))
                        { //If not, check if below is crosswalk, but only if this hit is road
                            isCrossWalk =
                                    isCrossWalk || ((nextHitActorName && strcasestr(nextHitActorName, "crosswalk")) ||
                                                    (nextNextHitActorName &&
                                                     strcasestr(nextNextHitActorName, "crosswalk")));
                        }
                    }

                    if (isCrossWalk)
                    {
                        label = SPECIAL_LABEL_CROSSWALKS;
                    }

                    // SPECIAL CASE BECAUSE DATA IS NOT SET CORRECTLY
                    const bool isUndefinedLabel = label == (int)CityObjectLabel::None || label == (int)CityObjectLabel::Static || label == (int)CityObjectLabel::Other;
                    if (isUndefinedLabel &&
                            strcasestr(actorHitName, "Road_Road") &&
                            (isCarlaLabelForStreetLevelObject(label) || isCarlaLabelForTerrainOrWater(label)))
                    {
                        label = (int)CityObjectLabel::Roads;
                    }
                    //--------------------------------------------------------------------

                    // When we hit the road, we put only one voxel in the grad to represent that.
                    if (isSideWalkOrRoadOrCrossWalk) // TODO: ASK MARIA !!! || isTerrainOrWaterHit)
                    {
                        // And that pos is the hit pos
                        actorHit_worldZStart = OutHit.Location.Z;
                        actorHit_worldZEnd = actorHit_worldZStart;
                    }

#if 0

                    // Correction step since we used the extended bounding box of the object
                    // From the hit point location go up abov8e with one ray until the current start and up to that point we CLEAR anything.
                    // this helps the things like below trees branches example
                    if (isSideWalkOrRoadOrCrossWalk || isTerrainOrWaterHit)
                    {
                        FVector raycastBackStart = OutHit.Location;
                        raycastBackStart.Z += 5.0f; // A little above the current hit to be sure that we don't hit it again
                        FVector raycastBackEnd = prevStart;
                        // Limit the raycastbackend to 2.4 m since nobody is tall enough like this to walk over those areas :)
                        const float limitAbove = 2000000.0f;
                        raycastBackEnd.Z = FMath::Min(raycastBackStart.Z + limitAbove, raycastBackEnd.Z);

                        FHitResult OutHitBack;
                        const bool isHitAbove = GetWorld()->LineTraceSingleByChannel(OutHitBack, raycastBackStart, raycastBackEnd, ECC_Visibility);

                        if (isHitAbove)
                        {
                            raycastBackEnd = OutHitBack.Location;
                        }
a
                        // Clear anything between [raycastStart voxel, raycastBackEnd)
                        const int raycast_backStartVoxelZ = (int)(raycastBackStart.Z / m_voxelSizeInCm);
                        const int raycast_backEndVoxelZ = (int)(raycastBackEnd.Z / m_voxelSizeInCm);
                        for (int voxelBackIndex = raycast_backStartVoxelZ; voxelBackIndex < raycast_backEndVoxelZ; voxelBackIndex++)
                        {
                            auto voxelPos = std::make_tuple(voxelX, voxelY, voxelBackIndex);

                            auto voxelIt = m_lastRaycastRes.find(voxelPos);
                            if (voxelIt != m_lastRaycastRes.end())
                            {
                                m_lastRaycastRes.erase(voxelIt);
                            }
                        }
                    }

#endif
                    // If not terrain then add to the map
                    if (!isSideWalkOrRoadOrCrossWalk && !isTerrainOrWaterHit)
                    {
                        // Add all static objects to a map to analyze later
                        AActor *hitActor = OutHit.GetActor();
                        if (allStaticObjectsMap.find(hitActor) == allStaticObjectsMap.end()) {
                            ObjectAndExtents extents;
                            extents.actor = hitActor;
                            extents.origin = actorHitOrigin;
                            extents.extent = actorHitExtent;
                            extents.label = label;
                            allStaticObjectsMap.insert(std::make_pair(hitActor, extents));
                        }
                    }

                    rayDebugInfoStep.isSideWalkOrRoadOrCrossWalk = isSideWalkOrRoadOrCrossWalk;
                    rayDebugInfoStep.isTerrainOrWaterHit = isTerrainOrWaterHit;

#if 0
                    // Mark all cells for raycast From prevStart to actorHit_worldZEnd mark all cells with label
                    if (isSideWalkOrRoadOrCrossWalk || isTerrainOrWaterHit)
#endif
                    {
                        const int prevStartZ_voxel = (int) (actorHit_worldZStart / m_voxelSizeInCm);
                        const int hitZEnd_voxel = (int) (actorHit_worldZEnd / m_voxelSizeInCm);

                        if (isDebugging && (prevStartZ_voxel < 0 || hitZEnd_voxel < 0))
                        {
                            // Check what happened !
                            for (int i = 0; i < debugInfo.steps.size(); i++)
                            {
                                const LastRayDebugInfoStep& debugStep = debugInfo.steps[i];
                                int a = 3;
                                a++;
                            }
                        }

                        PointCloudResult pcr;
                        pcr.rgbColor = debugRGBForActor;
                        pcr.segLabel = label;// prevLabel;
                        for (int iterZ = hitZEnd_voxel; iterZ <= prevStartZ_voxel; )
                        {
                            //if (iterZ == hitZEnd_voxel || iterZ == prevStartZ_voxel)
                            {
                                auto voxelPos = std::make_tuple(voxelX, voxelY, iterZ);
                                m_lastRaycastRes[voxelPos] = pcr;
                            }

                            iterZ += GetVoxelResolutionSkipStep(iterZ);

                            if (iterZ < prevStartZ_voxel)
                            {
                                iterZ = FMath::Min(iterZ, prevStartZ_voxel);
                            }
                            else
                            {
                                iterZ++;
                            }
                        }
                    }

                    // If street / terrain level hit here, we don't continue the search anymore, but we do a final step up to check how much space is free above ground
                    if (isSideWalkOrRoadOrCrossWalk || isTerrainOrWaterHit)
                    {
                        debug_wasGroundDiscovered = true;
                        continueSearch = false;

                        // Update the street level thing
                        const int currStartVoxel = (int)(currStart.Z / m_voxelSizeInCm);
                        if (currStartVoxel < m_lastRayCast_streetLevelMin)
                        {
                            m_lastRayCast_streetLevelMin = currStartVoxel;
                        }
                    }

                    prevLabel = label;
                }
                else
                {
                    if (!debug_wasGroundDiscovered)
                    {
                        UE_LOG(LogTemp, Error, TEXT("NOW i'm stopping !!  There was no ground detected for voxel (X,Y)=(%d,%d) coord (%f,%f):" ), voxelX, voxelY, topWorldVec.X, topWorldVec.Y);
                    }
                }

                //CityObjectLabel label
                //DrawDebugLine(GetWorld(), prevStart, endHit, hitColor, false, 1, 0, 1);
                prevStart = currStart;

                rayDebugInfoStep.label = label;
                rayDebugInfoStep.continueSearch = continueSearch;
                debugInfo.steps.push_back(rayDebugInfoStep);
            }

            if (!debug_wasGroundDiscovered && retries < MAX_RETRIES)
            {
                UE_LOG(LogTemp, Error, TEXT("Retry %d. There was no ground detected for voxel (X,Y)=(%d,%d) coord (%f,%f):" ), retries, voxelX, voxelY, topWorldVec.X, topWorldVec.Y);
                voxelY -= 1; // For debugging purposes
                retries++;
            }
            else
            {
                retries = 0;
            }
        }

        // Show some output stats
        if (voxelCellsEvaluated_slice > completionLogDebugRate)
        {
            voxelCellsEvaluated_total += voxelCellsEvaluated_slice;
            const float percentEvaluated = ((float) voxelCellsEvaluated_total / voxelCellsToEvaluate2D_total) * 100.0f;
            UE_LOG(LogTemp, Warning, TEXT("Percent %f:" ), percentEvaluated );
            voxelCellsEvaluated_slice = 0;
        }
    }

    // Analyze the static objects
# if 0
    for (auto& item : allStaticObjectsMap)
    {
        const AActor* key = item.first;
        const ObjectAndExtents& objectDef = item.second;
        const char* actorName = TCHAR_TO_ANSI(*objectDef.actor->GetName());

        // Ignore ground level things
        if (isCarlaLabelForStreetLevelObject(objectDef.label) || isCarlaLabelForTerrainOrWater(objectDef.label))
            continue;

        float collisionRadius = 0.0, collisionHeight = 0.0;
        key->GetSimpleCollisionCylinder(collisionRadius, collisionHeight);

        // Test with the object origin
        UPrimitiveComponent** outPrimitive1 = nullptr;
        FVector closestPointOnCollision1 = FVector::ZeroVector;
        const int res1 = key->ActorGetDistanceToCollision(objectDef.origin, ECollisionChannel::ECC_Visibility, closestPointOnCollision1, outPrimitive1);

        // Test with raycastActor location
        UPrimitiveComponent** outPrimitive2 = nullptr;
        FVector closestPointOnCollision2 = FVector::ZeroVector;
        const int res2 = key->ActorGetDistanceToCollision(centerWorld, ECollisionChannel::ECC_Visibility, closestPointOnCollision2, outPrimitive2);

        int res3 = res2;
        res3++;
    }
#endif

    // Keep only the margins of buildings, cars, street etc
    OptimizePointCloud();

    // Step 2: Write output the ply output file
    WriteOutputPlyFiles();
}

void ARayCastActor2::OptimizePointCloud()
{
    static int OPTIMIZATION_ENABLED = 0;

    if (OPTIMIZATION_ENABLED == 0)
    {
        return;
    }

    std::vector<std::tuple<int, int, int>> pointsToRemove; // Keep a data strcture of all points to remove
    pointsToRemove.reserve((int)(m_lastRaycastRes.size() * 0.75f));

    static int thresholdNumNeigbhs = 8;

    for (const auto& item : m_lastRaycastRes)
    {

        const int baseX = std::get<0>(item.first);
        const int baseY = std::get<1>(item.first);
        const int baseZ = std::get<2>(item.first);
        const int baseLabel = item.second.segLabel;

        if (isCarlaLabelForTerrainOrWater(baseLabel) || isCarlaLabelForStreetLevelObject(baseLabel))
            continue;

        // Cound number of neighbs
        int numNeighbs = 0;
        for (int devX = -1; devX <= 1; devX++)
        {
            for (int devY = -1; devY <= 1; devY++)
            {
                for (int devZ = -1; devZ <= 1; devZ++)
                {
                    const int absVal = std::abs(devX) + std::abs(devY) + std::abs(devZ);
                    if (absVal < 1)
                        continue;

                    const int testX = baseX + devX;
                    const int testY = baseY + devY;
                    const int testZ = baseZ + devZ;

                    const auto &it = m_lastRaycastRes.find(std::make_tuple(testX, testY, testZ));
                     if (it != m_lastRaycastRes.end())
                    {
                        const int testLabel = it->second.segLabel;
                        if (testLabel == baseLabel)
                        {
                            numNeighbs++;
                        }
                    }
                }
            }
        }

        // Is interior point ? Add to the remove list !
        if (numNeighbs >= thresholdNumNeigbhs)
        {
            pointsToRemove.emplace_back(item.first);
        }
    }

    UE_LOG(LogTemp, Warning, TEXT("There are %d points to remove"), pointsToRemove.size());
    const int initialSize = m_lastRaycastRes.size();
    for (const auto &point : pointsToRemove)
    {
        m_lastRaycastRes.erase(point);
    }
    const int currentSize = m_lastRaycastRes.size();
    UE_LOG(LogTemp, Warning, TEXT("After optimization out phase, from %d we got %d. Which is a %0.2f /% reduction"),
           initialSize, currentSize, ((initialSize - currentSize + 0.0f) / initialSize) * 100.0f);
}

void ARayCastActor2::FillMappings()
{
    // The belows mapping are taken from the ReconstructionUtils.py script from RLAgent code
    m_carlaToCitiscapesLabel.clear();
    m_carlaToCitiscapesLabel[0] = 0;  // None
    m_carlaToCitiscapesLabel[1] = 11;  // Building
    m_carlaToCitiscapesLabel[2] = 13 ; // Fence
    m_carlaToCitiscapesLabel[3] = 4;  // Other/Static
    m_carlaToCitiscapesLabel[4] = 24;  // Pedestrian
    m_carlaToCitiscapesLabel[5] = 17;  // Pole
    m_carlaToCitiscapesLabel[6] = 7;  // RoadLines
    m_carlaToCitiscapesLabel[7] = 7;  // Road
    m_carlaToCitiscapesLabel[8] = 8;  // Sidewalk
    m_carlaToCitiscapesLabel[9] = 21;  // Vegetation
    m_carlaToCitiscapesLabel[10] = 26;  // Vehicles
    m_carlaToCitiscapesLabel[11] = 12;  // Wall
    m_carlaToCitiscapesLabel[12] = 20;  // Traffic sign

    m_carlaToCitiscapesLabel[13] = 23 ;// Sky
    m_carlaToCitiscapesLabel[14] = 6 ;// Ground
    m_carlaToCitiscapesLabel[15] = 15 ;// Bridge
    m_carlaToCitiscapesLabel[16] = 10 ;// Railtrack
    m_carlaToCitiscapesLabel[17] = 14 ;// guardrail
    m_carlaToCitiscapesLabel[18] = 19 ;// traffic light
    m_carlaToCitiscapesLabel[19] = 4 ;// static
    m_carlaToCitiscapesLabel[20] = 5 ;// dynamic
    m_carlaToCitiscapesLabel[21] = 4 ;//water to s tatic..
    m_carlaToCitiscapesLabel[22] = 22 ;// terrain

    // Our special mappings / overridens
    m_carlaToCitiscapesLabel[(int)CityObjectLabel::Ground] = SPECIAL_LABEL_SIDEWALKS_EXTRA;
    m_carlaToCitiscapesLabel[SPECIAL_LABEL_CROSSWALKS] = SPECIAL_LABEL_CROSSWALKS;
    m_carlaToCitiscapesLabel[SPECIAL_LABEL_SIDEWALKS_EXTRA] = SPECIAL_LABEL_SIDEWALKS_EXTRA;

    m_cityscapes_segColorByLabel = std::vector<std::tuple<int,int,int>>{std::make_tuple(0, 0, 0), std::make_tuple(0, 0, 0), std::make_tuple(0, 0, 0), std::make_tuple(0, 0, 0), std::make_tuple(0, 0, 0),
            std::make_tuple(111, 74, 0), std::make_tuple(81, 0, 81), std::make_tuple(128, 64, 128), std::make_tuple(244, 35, 232), std::make_tuple(250, 170, 160),
            std::make_tuple(230, 150, 140), std::make_tuple(70, 70, 70), std::make_tuple(102, 102, 156), std::make_tuple(190, 153, 153), std::make_tuple(180, 165, 180),
            std::make_tuple(150, 100, 100), std::make_tuple(150, 120, 90), std::make_tuple(153, 153, 153), std::make_tuple(153, 153, 153), std::make_tuple(250, 170, 30),
            std::make_tuple(220, 220, 0), std::make_tuple(107, 142, 35), std::make_tuple(152, 251, 152), std::make_tuple(70, 130, 180), std::make_tuple(220, 20, 60),
            std::make_tuple(255, 0, 0), std::make_tuple(0, 0, 142), std::make_tuple(0, 0, 70), std::make_tuple(0, 60, 100), std::make_tuple(0, 0, 90),
            std::make_tuple(0, 0, 110), std::make_tuple(0, 80, 100), std::make_tuple(0, 0, 230), std::make_tuple(119, 11, 32),
            std::make_tuple(255, 255, 255), std::make_tuple(255, 255, 0)      // The special ones crosswalks then sidewalks extra
            };
}

int ARayCastActor2::getCarlaToCityscapesLabel(int carlaLabel)
{
    return m_carlaToCitiscapesLabel[carlaLabel];
}

std::tuple<int, int, int> ARayCastActor2::getCityscapesSegColorOfLabel(int cityscapesLabel)
{
    return m_cityscapes_segColorByLabel[cityscapesLabel];
}

// Called every frame
void ARayCastActor2::Tick(float DeltaTime)
{
	AActor::Tick(DeltaTime);
    DoSetup();

	static bool didRayCast = false;
	if (g_restartDebug)
    {
	    didRayCast = false;
	    g_restartDebug = false;
    }

	if (!didRayCast)
    {
        didRayCast = true;
        //PerformRaycast();
    }
}

#pragma GCC pop_options