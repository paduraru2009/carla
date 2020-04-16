// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "WalkerControl.generated.h"

// TODO: get rid of this
#define NUM_PFNN_BONES 31

USTRUCT(BlueprintType)
struct CARLA_API FWalkerControl
{
  GENERATED_BODY()

  UPROPERTY(Category = "Walker Control", EditAnywhere, BlueprintReadWrite)
  FVector Direction = {1.0f, 0.0f, 0.0f};

  UPROPERTY(Category = "Walker Control", EditAnywhere, BlueprintReadWrite)
  float Speed = 0.0f;

  UPROPERTY(Category = "Walker Control", EditAnywhere, BlueprintReadWrite)
  bool Jump = false;

    UPROPERTY(Category = "Walker Control", EditAnywhere, BlueprintReadWrite)
    bool forceTargetPosition = false;

  UPROPERTY(Category = "Walker PFNN Control", EditAnywhere, BlueprintReadWrite)
  bool usePFNNInference; // Should it use inference or we receive the skeleton already computed ?

  UPROPERTY(Category = "Walker PFNN Control", EditAnywhere, BlueprintReadWrite)
  bool usePFNN; // if should use PFNN at all or just Carla animation system

  UPROPERTY(Category = "Walker PFNN Control", EditAnywhere, BlueprintReadWrite)
  FVector nextTargetPos = {0.0f, 0.0f, 0.0f};

  UPROPERTY(Category = "Walker PFNN Control", EditAnywhere, BlueprintReadWrite)
  FVector futureTargetPos = {0.0f, 0.0f, 0.0f};

  //UPROPERTY(Category = "Walker PFNN Control", EditAnywhere, BlueprintReadWrite)
  // A position vector for each of the bones
  float poses[NUM_PFNN_BONES * 3];

};
