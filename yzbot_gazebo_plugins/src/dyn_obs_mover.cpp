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
    if (!this->obs3_) {
      this->obs3_ = this->world_->ModelByName("dyn_obs_3");
    }
    if (!this->obs1_ || !this->obs2_) {
      return;
    }

    // obs1 / obs2: slide on X. obs3: slide on Y.
    // obs1（上面）：x ∈ [-2.75, 1.15], period 31s
    const double x1 = -0.80 + 1.95 * std::sin(2.0 * M_PI * t / 31.0);
    SetPose(this->obs1_, x1, -1.35, 0.20);

    // obs2（下面）：x ∈ [-2.30, 1.00], period 33.67s（负向再伸 1m）
    const double x2 = -0.65 + 1.65 * std::sin(2.0 * M_PI * t / 33.67 + 1.2);
    SetPose(this->obs2_, x2, -6.15, 0.20);

    // obs3: fixed x=-2, y ∈ [-2.5, 2.5], period 45s
    if (this->obs3_) {
      const double y3 = 0.0 + 2.5 * std::sin(2.0 * M_PI * t / 45.0 + 2.4);
      SetPose(this->obs3_, -2.0, y3, 0.20);
    }
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
  gazebo::physics::ModelPtr obs3_;
  gazebo::event::ConnectionPtr update_connection_;
  double last_t_{-1.0};
};

GZ_REGISTER_WORLD_PLUGIN(DynObsMover)

}  // namespace yzbot
