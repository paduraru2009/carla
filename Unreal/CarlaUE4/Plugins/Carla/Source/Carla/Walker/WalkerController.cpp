// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla.h"
#include "Carla/Walker/WalkerController.h"

#include "Components/PoseableMeshComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Containers/Map.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Pawn.h"

#include <boost/variant/apply_visitor.hpp>

#define CM_TO_METER		0.01f
#define M_TO_CENTIMETERS 100.0f

AWalkerController::AWalkerController(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
  PrimaryActorTick.bCanEverTick = true;
}

void AWalkerController::OnPossess(APawn *InPawn)
{
  Super::OnPossess(InPawn);

  auto *Character = Cast<ACharacter>(InPawn);
  if (Character == nullptr)
  {
    UE_LOG(LogCarla, Error, TEXT("Walker is not a character!"));
    return;
  }

  auto *MovementComponent = Character->GetCharacterMovement();
  if (MovementComponent == nullptr)
  {
    UE_LOG(LogCarla, Error, TEXT("Walker missing character movement component!"));
    return;
  }

  MovementComponent->MaxWalkSpeed = GetMaximumWalkSpeed();
  MovementComponent->JumpZVelocity = 500.0f;
  Character->JumpMaxCount = 2;
}

UPoseableMeshComponent *AddNewBoneComponent(AActor *InActor, FVector inLocation, FRotator inRotator)
{
  UPoseableMeshComponent *NewComp = NewObject<UPoseableMeshComponent>(InActor,
      UPoseableMeshComponent::StaticClass());
  if (NewComp)
  {
    NewComp->RegisterComponent();
    NewComp->SetWorldLocation(inLocation);
    NewComp->SetWorldRotation(inRotator);
    NewComp->AttachToComponent(InActor->GetRootComponent(), FAttachmentTransformRules::KeepRelativeTransform);
  }
  return NewComp;
}

void AWalkerController::ApplyWalkerControl(const FWalkerControl &InControl)
{
  Control = InControl;
  if (bManualBones)
  {
    SetManualBones(false);
  }
}

void AWalkerController::ApplyWalkerControl(const FWalkerBoneControl &InBoneControl)
{
  Control = InBoneControl;
  if (!bManualBones)
  {
    SetManualBones(true);
  }
}

void AWalkerController::SetManualBones(const bool bIsEnabled)
{
  bManualBones = bIsEnabled;

  auto *Character = GetCharacter();
  TArray<UPoseableMeshComponent *> PoseableMeshes;
  TArray<USkeletalMeshComponent *> SkeletalMeshes;
  Character->GetComponents<UPoseableMeshComponent>(PoseableMeshes, false);
  Character->GetComponents<USkeletalMeshComponent>(SkeletalMeshes, false);
  USkeletalMeshComponent *SkeletalMesh = SkeletalMeshes.IsValidIndex(0) ? SkeletalMeshes[0] : nullptr;
  if (SkeletalMesh)
  {
    if (bManualBones)
    {
      UPoseableMeshComponent *PoseableMesh =
          PoseableMeshes.IsValidIndex(0) ? PoseableMeshes[0] : AddNewBoneComponent(Character,
          SkeletalMesh->GetRelativeTransform().GetLocation(),
          SkeletalMesh->GetRelativeTransform().GetRotation().Rotator());
      PoseableMesh->SetSkeletalMesh(SkeletalMesh->SkeletalMesh);
      PoseableMesh->SetVisibility(true);
      SkeletalMesh->SetVisibility(false);
    }
    else
    {
      UPoseableMeshComponent *PoseableMesh = PoseableMeshes.IsValidIndex(0) ? PoseableMeshes[0] : nullptr;
      PoseableMesh->SetVisibility(false);
      SkeletalMesh->SetVisibility(true);
    }
  }
}

void AWalkerController::ControlTickVisitor::operator()(FWalkerControl &WalkerControl)
{
  auto *Character = Controller->GetCharacter();
  if (Character != nullptr)
  {
    // Do we have a forced target to go in the control ?
  	if (WalkerControl.forceTargetPosition)
	{
      const FVector currentPos = Character->GetActorLocation();

      const FVector forward = Character->GetActorForwardVector();

      // Convert positions to centimeters :
      // TODO: convert them at receive time
      const FVector targetPos_cm = WalkerControl.nextTargetPos * M_TO_CENTIMETERS;
      for (int i = 0; i < NUM_PFNN_BONES * 3; i++)
      {
          WalkerControl.poses[i] *= M_TO_CENTIMETERS;
      }

      FVector direction = targetPos_cm - currentPos;
      direction.Z = 0.0f;

      const float maxSpeed = Controller->GetMaximumWalkSpeed();
      const float targetSpeedPercent = WalkerControl.Speed / maxSpeed;

      const bool useControlDirection = true;
      Character->AddMovementInput(useControlDirection ? WalkerControl.Direction : direction, targetSpeedPercent);

      FVector newPos = Character->GetActorLocation();
      
      // TODO: check overshoot

      // Render skeleton target if requested
      if (WalkerControl.usePFNN)
      {
          int parents[NUM_PFNN_BONES] = {-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27};

          for (int i = 0; i < NUM_PFNN_BONES; i++)
          {
              const float* jointPosBase = &WalkerControl.poses[i*3];
              FVector jointPos(jointPosBase[0], jointPosBase[1], jointPosBase[2]);
              //jointPos += targetPos_cm; // Add the ideal agent pos

              int parent = parents[i];
              if (parent != -1)
              {
                  const float* jointParentPoseBase = &WalkerControl.poses[parent*3];
                  FVector jointParentPos(jointParentPoseBase[0], jointParentPoseBase[1], jointParentPoseBase[2]);
                  //jointParentPos += targetPos_cm; // Add the ideal agent pos

                  DrawDebugLine(Character->GetWorld(), jointParentPos, jointPos, FColor::Emerald, false, -1.0f, ECC_WorldStatic, 3.f);
              }
          }
      }

      // TODO: do pfnn inference !
  	}
	else
	{
	    Character->AddMovementInput(WalkerControl.Direction,
	        WalkerControl.Speed / Controller->GetMaximumWalkSpeed());
	    if (WalkerControl.Jump)
	    {
	      Character->Jump();
	    }
	}
  }
}

void AWalkerController::ControlTickVisitor::operator()(FWalkerBoneControl &WalkerBoneControl)
{
  auto *Character = Controller->GetCharacter();
  if (Character == nullptr)
  {
    return;
  }
  TArray<UPoseableMeshComponent *> PoseableMeshes;
  Character->GetComponents<UPoseableMeshComponent>(PoseableMeshes, false);
  UPoseableMeshComponent *PoseableMesh = PoseableMeshes.IsValidIndex(0) ? PoseableMeshes[0] : nullptr;
  if (PoseableMesh)
  {
    for (const TPair<FString, FTransform> &pair : WalkerBoneControl.BoneTransforms)
    {
      FName BoneName = FName(*pair.Key);
      PoseableMesh->SetBoneTransformByName(BoneName, pair.Value, EBoneSpaces::Type::ComponentSpace);
    }
    WalkerBoneControl.BoneTransforms.Empty();
  }
}

void AWalkerController::Tick(float DeltaSeconds)
{
  Super::Tick(DeltaSeconds);
  AWalkerController::ControlTickVisitor ControlTickVisitor(this);
  boost::apply_visitor(ControlTickVisitor, Control);
}
