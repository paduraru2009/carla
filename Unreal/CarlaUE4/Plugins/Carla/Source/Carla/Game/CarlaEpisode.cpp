// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"

#include <compiler/disable-ue4-macros.h>
#include <carla/opendrive/OpenDriveParser.h>
#include <carla/rpc/String.h>
#include <compiler/enable-ue4-macros.h>

#include "Carla/Sensor/Sensor.h"
#include "Carla/Util/BoundingBoxCalculator.h"
#include "Carla/Util/RandomEngine.h"
#include "Carla/Vehicle/VehicleSpawnPoint.h"
#include "Carla/Game/CarlaStatics.h"

#include "Engine/StaticMeshActor.h"
#include "EngineUtils.h"
#include "GameFramework/SpectatorPawn.h"
#include "GenericPlatform/GenericPlatformProcess.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Math/Vector.h"
#include "Carla/Game/RayCastActor2.h"

static FString UCarlaEpisode_GetTrafficSignId(ETrafficSignState State)
{
  using TSS = ETrafficSignState;
  switch (State)
  {
    case TSS::TrafficLightRed:
    case TSS::TrafficLightYellow:
    case TSS::TrafficLightGreen:  return TEXT("traffic.traffic_light");
    case TSS::SpeedLimit_30:      return TEXT("traffic.speed_limit.30");
    case TSS::SpeedLimit_40:      return TEXT("traffic.speed_limit.40");
    case TSS::SpeedLimit_50:      return TEXT("traffic.speed_limit.50");
    case TSS::SpeedLimit_60:      return TEXT("traffic.speed_limit.60");
    case TSS::SpeedLimit_90:      return TEXT("traffic.speed_limit.90");
    case TSS::SpeedLimit_100:     return TEXT("traffic.speed_limit.100");
    case TSS::SpeedLimit_120:     return TEXT("traffic.speed_limit.120");
    case TSS::SpeedLimit_130:     return TEXT("traffic.speed_limit.130");
    case TSS::StopSign:           return TEXT("traffic.stop");
    case TSS::YieldSign:          return TEXT("traffic.yield");
    default:                      return TEXT("traffic.unknown");
  }
}

UCarlaEpisode::UCarlaEpisode(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer),
    Id(URandomEngine::GenerateRandomId())
{
  ActorDispatcher = CreateDefaultSubobject<UActorDispatcher>(TEXT("ActorDispatcher"));
}

bool UCarlaEpisode::LoadNewEpisode(const FString &MapString)
{
  FString FinalPath = MapString.IsEmpty() ? GetMapName() : MapString;
  bool bIsFileFound = false;
  if (MapString.StartsWith("/Game"))
  {
    // Full path
    if (!MapString.EndsWith(".umap"))
    {
      FinalPath += ".umap";
    }
    // Some conversions...
    FinalPath = FinalPath.Replace(TEXT("/Game/"), *FPaths::ProjectContentDir());
    if (FPaths::FileExists(IFileManager::Get().ConvertToAbsolutePathForExternalAppForRead(*FinalPath)))
    {
      bIsFileFound = true;
      FinalPath = MapString;
    }
  }
  else
  {
    if (MapString.Contains("/"))
    {
      bIsFileFound = false;
    }
    else
    {
      // Find the full path under Carla
      TArray<FString> TempStrArray, PathList;
      if (!MapString.EndsWith(".umap"))
      {
        FinalPath += ".umap";
      }
      IFileManager::Get().FindFilesRecursive(PathList, *FPaths::ProjectContentDir(), *FinalPath, true, false, false);
      if (PathList.Num() > 0)
      {
        FinalPath = PathList[0];
        FinalPath.ParseIntoArray(TempStrArray, TEXT("Content/"), true);
        FinalPath = TempStrArray[1];
        FinalPath.ParseIntoArray(TempStrArray, TEXT("."), true);
        FinalPath = "/Game/" + TempStrArray[0];
        bIsFileFound = true;
      }
    }
  }
  if (bIsFileFound)
  {
    UE_LOG(LogCarla, Warning, TEXT("Loading a new episode: %s"), *FinalPath);
    UGameplayStatics::OpenLevel(GetWorld(), *FinalPath, true);
    ApplySettings(FEpisodeSettings{});
  }
  return bIsFileFound;
}

static FString BuildRecastBuilderFile()
{
  // Define filename with extension depending on if we are on Windows or not
#if PLATFORM_WINDOWS
  const FString RecastToolName = "RecastBuilder.exe";
#else
  const FString RecastToolName = "RecastBuilder";
#endif // PLATFORM_WINDOWS

  // Define path depending on the UE4 build type (Package or Editor)
#if UE_BUILD_SHIPPING
  const FString AbsoluteRecastBuilderPath = FPaths::ConvertRelativePathToFull(
      FPaths::RootDir() + "Tools/" + RecastToolName);
#else
  const FString AbsoluteRecastBuilderPath = FPaths::ConvertRelativePathToFull(
      FPaths::ProjectDir() + "../../Util/DockerUtils/dist/" + RecastToolName);
#endif
  return AbsoluteRecastBuilderPath;
}

bool UCarlaEpisode::LoadNewOpendriveEpisode(
    const FString &OpenDriveString,
    const carla::rpc::OpendriveGenerationParameters &Params)
{
  if (OpenDriveString.IsEmpty())
  {
    UE_LOG(LogCarla, Error, TEXT("The OpenDrive string is empty."));
    return false;
  }

  // Build the Map from the OpenDRIVE data
  const auto CarlaMap = carla::opendrive::OpenDriveParser::Load(
      carla::rpc::FromLongFString(OpenDriveString));

  // Check the Map is correclty generated
  if (!CarlaMap.has_value())
  {
    UE_LOG(LogCarla, Error, TEXT("The OpenDrive string is invalid or not supported"));
    return false;
  }

  // Generate the OBJ (as string)
  const auto RoadMesh = CarlaMap->GenerateMesh(Params.vertex_distance);
  const auto CrosswalksMesh = CarlaMap->GetAllCrosswalkMesh();
  const auto RecastOBJ = (RoadMesh + CrosswalksMesh).GenerateOBJForRecast();

  const FString AbsoluteOBJPath = FPaths::ConvertRelativePathToFull(
      FPaths::ProjectContentDir() + "Carla/Maps/Nav/OpenDriveMap.obj");

  // Store the OBJ string to a file in order to that RecastBuilder can load it
  FFileHelper::SaveStringToFile(
      carla::rpc::ToLongFString(RecastOBJ),
      *AbsoluteOBJPath,
      FFileHelper::EEncodingOptions::ForceUTF8,
      &IFileManager::Get());

  const FString AbsoluteXODRPath = FPaths::ConvertRelativePathToFull(
      FPaths::ProjectContentDir() + "Carla/Maps/OpenDrive/OpenDriveMap.xodr");

  // Copy the OpenDrive as a file in the serverside
  FFileHelper::SaveStringToFile(
      OpenDriveString,
      *AbsoluteXODRPath,
      FFileHelper::EEncodingOptions::ForceUTF8,
      &IFileManager::Get());

  if (!FPaths::FileExists(AbsoluteXODRPath))
  {
    UE_LOG(LogCarla, Error, TEXT("ERROR: XODR not copied!"));
    return false;
  }

  UCarlaGameInstance * GameInstance = UCarlaStatics::GetGameInstance(GetWorld());
  if(GameInstance)
  {
    GameInstance->SetOpendriveGenerationParameters(Params);
  }
  else
  {
    carla::log_warning("Missing game instance");
  }

  const FString AbsoluteRecastBuilderPath = BuildRecastBuilderFile();

  if (FPaths::FileExists(AbsoluteRecastBuilderPath) &&
      Params.enable_pedestrian_navigation)
  {
    /// @todo this can take too long to finish, clients need a method
    /// to know if the navigation is available or not.
    FPlatformProcess::CreateProc(
        *AbsoluteRecastBuilderPath, *AbsoluteOBJPath,
        true, true, true, nullptr, 0, nullptr, nullptr);
  }
  else
  {
    UE_LOG(LogCarla, Warning, TEXT("'RecastBuilder' not present under '%s', "
        "the binaries for pedestrian navigation will not be created."),
        *AbsoluteRecastBuilderPath);
  }

  return true;
}

void UCarlaEpisode::ApplySettings(const FEpisodeSettings &Settings)
{
  FCarlaStaticDelegates::OnEpisodeSettingsChange.Broadcast(Settings);
  EpisodeSettings = Settings;
}

TArray<FTransform> UCarlaEpisode::GetRecommendedSpawnPoints() const
{
  TArray<FTransform> SpawnPoints;
  for (TActorIterator<AVehicleSpawnPoint> It(GetWorld()); It; ++It)
  {
    SpawnPoints.Add(It->GetActorTransform());
  }
  return SpawnPoints;
}

//#pragma optimize("", off)

void UCarlaEpisode::GetSpawnPointsNearCrossWalks(std::vector< UCarlaEpisode::IndexAndSpawnTransform >& outSpawnPoints) const
{
    outSpawnPoints.clear();

    const float MAX_DIST_TO_CROSSWALK_CM = (20.0f * 100.0f); // Centimeters
    const float MIN_DIST_TO_CROSSWALK_CM = (7.0f * 100.0f); // Centimeters
    const float MAX_DIST_TO_CROSSWALK_CM_SQR = MAX_DIST_TO_CROSSWALK_CM * MAX_DIST_TO_CROSSWALK_CM;
    const float MIN_DIST_TO_CROSSWALK_CM_SQR = MIN_DIST_TO_CROSSWALK_CM * MIN_DIST_TO_CROSSWALK_CM;
    const float MAX_DIST_TO_CROSSWALK_HEIGHT_CM = (1.0f * 200.0f); // Cm
	const float MAX_ANGLE_TO_CROSSWALK_DEG = 45; // 45 degrees orientation to a crosswalk maximum
	
	// Get the transforms of all crosswalks
    TArray<FTransform> CrossWalkTransforms;
    UWorld* world = GetWorld();
	for (TActorIterator<AStaticMeshActor> It(world); It; ++It)
	{
        const FString& str = It->GetName();
        if (str.Contains("Road_Crosswalk"))
        {
            CrossWalkTransforms.Add(It->GetTransform());        	
        }		
	}

	// Iterate over all spawnpoints and select only those that are close to the crosswalks
	// And oriented towards them    
    int index = 0;
    for (TActorIterator<AVehicleSpawnPoint> It(GetWorld()); It; ++It, ++index)
    {
        const FTransform& spawnPointTransform = It->GetActorTransform();

    	// Check this against all crosswalk
    	// TODO: optimize with a tree space partitioning..
        bool foundCrossWalkInFrontAndClose = false;
		for (const FTransform& crossWalkTransform : CrossWalkTransforms)
		{
			// Check dist
            FVector spawnPointToCrossWalk = crossWalkTransform.GetLocation() - spawnPointTransform.GetLocation();
            const float distance = spawnPointToCrossWalk.SizeSquared();
            if (MIN_DIST_TO_CROSSWALK_CM_SQR > distance || distance > MAX_DIST_TO_CROSSWALK_CM_SQR)
                continue;
             
			if (FMath::Abs(spawnPointToCrossWalk.Z) > MAX_DIST_TO_CROSSWALK_HEIGHT_CM)
			{
                continue;
			}

            spawnPointToCrossWalk.Z = 0.0f;

			// Check orientation
            const FVector spawnPointForward = spawnPointTransform.GetRotation().GetForwardVector().GetSafeNormal();
            const FVector spawnPointToCrossWalkNorm = spawnPointToCrossWalk.GetSafeNormal();
            const float dotBetweenVectors = FVector::DotProduct(spawnPointForward, spawnPointToCrossWalkNorm);
            const float angleInRads = FMath::RadiansToDegrees(FMath::Acos(dotBetweenVectors));
            if (angleInRads > MAX_ANGLE_TO_CROSSWALK_DEG)
                continue;

            foundCrossWalkInFrontAndClose = true;
            break;
		}

    	if (foundCrossWalkInFrontAndClose)
    	{
            outSpawnPoints.push_back(std::make_pair(index, spawnPointTransform));
    	}
    }
}

//#pragma optimize("", on)


carla::rpc::Actor UCarlaEpisode::SerializeActor(FActorView ActorView) const
{
  carla::rpc::Actor Actor;
  if (ActorView.IsValid())
  {
    Actor = ActorView.GetActorInfo()->SerializedData;
    auto Parent = ActorView.GetActor()->GetOwner();
    if (Parent != nullptr)
    {
      Actor.parent_id = FindActor(Parent).GetActorId();
    }
  }
  else
  {
    UE_LOG(LogCarla, Warning, TEXT("Trying to serialize invalid actor"));
  }
  return Actor;
}

void UCarlaEpisode::AttachActors(
    AActor *Child,
    AActor *Parent,
    EAttachmentType InAttachmentType)
{
  UActorAttacher::AttachActors(Child, Parent, InAttachmentType);

  // recorder event
  if (Recorder->IsEnabled())
  {
    CarlaRecorderEventParent RecEvent
    {
      FindActor(Child).GetActorId(),
      FindActor(Parent).GetActorId()
    };
    Recorder->AddEvent(std::move(RecEvent));
  }
}


//#pragma GCC push_options
//#pragma GCC optimize ("O0")

void UCarlaEpisode::InitializeAtBeginPlay() {
    auto World = GetWorld();
    check(World != nullptr);
    auto PlayerController = UGameplayStatics::GetPlayerController(World, 0);
    if (PlayerController == nullptr) {
        UE_LOG(LogCarla, Error, TEXT("Can't find player controller!"));
        return;
    }
    Spectator = PlayerController->GetPawn();
    if (Spectator != nullptr) {
        FActorDescription Description;
        Description.Id = TEXT("spectator");
        Description.Class = Spectator->GetClass();
        ActorDispatcher->RegisterActor(*Spectator, Description);
    } else {
        UE_LOG(LogCarla, Error, TEXT("Can't find spectator!"));
    }

    RaycastActor = nullptr;
    TArray < ARayCastActor2 * > raycastActors;
    for (TActorIterator <ARayCastActor2> It(World); It; ++It) {
        raycastActors.Add(*It);
    }

    if (raycastActors.Num() > 0) {
        ensureMsgf(raycastActors.Num() <= 1,
                   TEXT("There are more than 1 raycast actors ! I will select the first one"));
        RaycastActor = raycastActors[0];
    }

    if (RaycastActor != nullptr) {
        FActorDescription Description;
        Description.Id = TEXT("raycastActor");
        Description.Class = RaycastActor->GetClass();
        ActorDispatcher->RegisterActor(*RaycastActor, Description);
    }

    auto ActorView = FindActor(GetRaycastActor());
    if (!ActorView.IsValid())
    {
        int a = 3;
        a++;
    }

  for (TActorIterator<ATrafficSignBase> It(World); It; ++It)
  {
    ATrafficSignBase *Actor = *It;
    check(Actor != nullptr);
    FActorDescription Description;
    Description.Id = UCarlaEpisode_GetTrafficSignId(Actor->GetTrafficSignState());
    Description.Class = Actor->GetClass();
    ActorDispatcher->RegisterActor(*Actor, Description);
  }

  // get the definition id for static.prop.mesh
  auto Definitions = GetActorDefinitions();
  uint32 StaticMeshUId = 0;
  for (auto& Definition : Definitions)
  {
    if (Definition.Id == "static.prop.mesh")
    {
      StaticMeshUId = Definition.UId;
      break;
    }
  }

  for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
  {
    auto Actor = *It;
    check(Actor != nullptr);
    auto MeshComponent = Actor->GetStaticMeshComponent();
    check(MeshComponent != nullptr);
    if (MeshComponent->Mobility == EComponentMobility::Movable)
    {
      FActorDescription Description;
      Description.Id = TEXT("static.prop.mesh");
      Description.UId = StaticMeshUId;
      Description.Class = Actor->GetClass();
      Description.Variations.Add("mesh_path",
          FActorAttribute{"mesh_path", EActorAttributeType::String,
          MeshComponent->GetStaticMesh()->GetPathName()});
      Description.Variations.Add("mass",
          FActorAttribute{"mass", EActorAttributeType::Float,
          FString::SanitizeFloat(MeshComponent->GetMass())});
      ActorDispatcher->RegisterActor(*Actor, Description);
    }
  }
}

//#pragma GCC pop_options

void UCarlaEpisode::CaptureRaycastActor(std::string outPath, const bool synchronous)
{
    RaycastActor->m_pathToOutput = FString(outPath.c_str());
    UE_LOG(LogTemp, Warning, TEXT("!!! Starting raycast and OUTPUT to %s" ), *RaycastActor->m_pathToOutput);
    RaycastActor->PerformRaycast(synchronous);
}

void UCarlaEpisode::EndPlay(void)
{
  // stop recorder and replayer
  if (Recorder)
  {
    Recorder->Stop();
    if (Recorder->GetReplayer()->IsEnabled())
    {
      Recorder->GetReplayer()->Stop();
    }
  }
}

std::string UCarlaEpisode::StartRecorder(std::string Name, bool AdditionalData)
{
  std::string result;

  if (Recorder)
  {
    result = Recorder->Start(Name, MapName, AdditionalData);
  }
  else
  {
    result = "Recorder is not ready";
  }

  return result;
}
