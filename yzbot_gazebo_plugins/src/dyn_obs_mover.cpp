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

    // Orange path (now grey): straight N-S at x=-1.3; period 40s
    const double y1 = 1.6 * std::sin(2.0 * M_PI * t / 40.0);
    SetPose(this->obs1_, -1.3, y1, 0.20);

    // Cyan path (now grey): straight E-W at y=-6.5; center x≈0.5
    const double x2 = 0.5 + 1.2 * std::sin(2.0 * M_PI * t / 40.0);
    SetPose(this->obs2_, x2, -6.5, 0.20);
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
