// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"

#ifdef LIBCARLA_INCLUDED_FROM_UE4
#  include "Carla/Walker/WalkerControl.h"
#endif // LIBCARLA_INCLUDED_FROM_UE4

#include <vector>

namespace carla {
namespace rpc {

  class WalkerControl {
  public:

    WalkerControl() = default;

   WalkerControl(
            geom::Vector3D in_direction,
            float in_speed,
            bool in_jump,
            bool in_use_forced_target,
            geom::Vector3D in_target_pos,
            bool in_usePFNN)
            //,std::vector<float>& in_poses)
            : direction(in_direction),
              speed(in_speed),
              jump(in_jump),
              useForcedTarget(in_use_forced_target),
              targetPos(in_target_pos), 
              usePFNN(in_usePFNN)
              //poses(in_poses)
              {
                poses.clear();
              }

   WalkerControl(
        geom::Vector3D in_direction,
        float in_speed,
        bool in_jump)
        //,std::vector<float>& in_poses)
        : direction(in_direction),
          speed(in_speed),
          jump(in_jump),
          useForcedTarget(false),
          targetPos({1.0f, 0.0f, 0.0f}), 
          usePFNN(false)
          //poses(in_poses)
          {
            poses.clear();
          }


      geom::Vector3D direction = {1.0f, 0.0f, 0.0f};
      float speed = 0.0f;
      bool jump = false;
      bool useForcedTarget = false;
      geom::Vector3D targetPos = {0.0f, 0.0f, 0.0f};
      bool usePFNN = false;
      std::vector<float> poses;

#ifdef LIBCARLA_INCLUDED_FROM_UE4

      WalkerControl(const FWalkerControl& Control)
          : direction(Control.Direction.X, Control.Direction.Y, Control.Direction.Z),
          speed(1e-2f * Control.Speed),
          jump(Control.Jump),
          useForcedTarget(Control.forceTargetPosition)
      {
          targetPos = geom::Vector3D(Control.nextTargetPos.X, Control.nextTargetPos.Y, Control.nextTargetPos.Z);
          usePFNN = Control.usePFNN;

          // TODO: memcpy
          if (usePFNN)
          {
              poses.clear();
              poses.resize(NUM_PFNN_BONES * 3);
              for (int i = 0; i < NUM_PFNN_BONES * 3; i++)
              {
                  poses[i] = Control.poses[i];
              }
          }
      }


      operator FWalkerControl() const {
          FWalkerControl Control;
          Control.Direction = { direction.x, direction.y, direction.z };
          Control.Speed = 1e2f * speed;
          Control.Jump = jump;
          Control.forceTargetPosition = useForcedTarget;
          Control.nextTargetPos = targetPos.ToFVector();


          Control.usePFNN = usePFNN;

          if (usePFNN)
          {
              // TODO: memcpy
              assert(poses.size() == NUM_PFNN_BONES * 3);
              for (int i = 0; i < NUM_PFNN_BONES * 3; i++)
              {
                  Control.poses[i] = poses[i];
              }
          }

          return Control;
      }

#endif // LIBCARLA_INCLUDED_FROM_UE4

    bool operator!=(const WalkerControl &rhs) const {
      return direction != rhs.direction || speed != rhs.speed || jump != rhs.jump || poses != rhs.poses;
    }

    bool operator==(const WalkerControl &rhs) const {
      return !(*this != rhs);
    }

    MSGPACK_DEFINE_ARRAY(direction, speed, jump, useForcedTarget, targetPos, usePFNN, poses); // poses, usePFNN);
  };

} // namespace rpc
} // namespace carla

