// In-process dynamic obstacle mover for Gazebo classic.
// Throttled ~5 Hz (not every physics step) to save CPU/RAM.

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/common/Events.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>

#include <cmath>
#include <string>

namespace yzbot
{

class DynObsMover : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr /*sdf*/) override
  {
    this->world_ = world;
    if (this->world_->IsPaused()) {
      this->world_->SetPaused(false);
      gzmsg << "[dyn_obs_mover] was paused — unpaused\n";
    }
    this->last_t_ = -1.0;
    this->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&DynObsMover::OnUpdate, this));
    gzmsg << "[dyn_obs_mover] loaded @~5Hz — grey kinematic movers (laser only)\n";
  }

private:
  void OnUpdate()
  {
    if (!this->world_) {
      return;
    }

    const double t = this->world_->SimTime().Double();
    // Throttle: ~5 Hz pose updates (saves CPU; motion still looks smooth enough)
    if (this->last_t_ >= 0.0 && (t - this->last_t_) < 0.20) {
      return;
    }
    this->last_t_ = t;

    if (!this->obs1_) {
      this->obs1_ = this->world_->ModelByName("dyn_obs_1");
    }
    if (!this->obs2_) {
      this->obs2_ = this->world_->ModelByName("dyn_obs_2");
    }
    if (!this->obs1_ || !this->obs2_) {
      return;
    }

    // Both horizontal bars slide on X. Keep clear of cube lanes.
    // Period ~42s; smaller amp + mid-corridor Y so bars don't skim cubes.
    // obs1: between blue#3 (y≈-0.4) and zone B (y≈-4)
    const double x1 = 0.20 + 0.50 * std::sin(2.0 * M_PI * t / 42.0);
    SetPose(this->obs1_, x1, -1.35, 0.20);

    // obs2: keep clear of blue#5 (south) and red#3/#4 (north); smaller amp
    const double x2 = 0.00 + 0.45 * std::sin(2.0 * M_PI * t / 42.0 + 1.2);
    SetPose(this->obs2_, x2, -6.15, 0.20);
  }

  static void SetPose(
    const gazebo::physics::ModelPtr & model, double x, double y, double z)
  {
    ignition::math::Pose3d pose(x, y, z, 0, 0, 0);
    model->SetWorldPose(pose);
    model->SetLinearVel(ignition::math::Vector3d::Zero);
    model->SetAngularVel(ignition::math::Vector3d::Zero);
  }

  gazebo::physics::WorldPtr world_;
  gazebo::physics::ModelPtr obs1_;
  gazebo::physics::ModelPtr obs2_;
  gazebo::event::ConnectionPtr update_connection_;
  double last_t_{-1.0};
};

GZ_REGISTER_WORLD_PLUGIN(DynObsMover)

}  // namespace yzbot
