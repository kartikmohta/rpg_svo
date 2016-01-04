// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
// Copyright (C) 2015 Kartik Mohta <kartikmohta@gmail.com>
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/config.h>

namespace svo
{
int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

Seed::Seed(Feature *ftr, float depth_mean, float depth_min)
    : batch_id(batch_counter),
      id(seed_counter++),
      ftr(ftr),
      a(10),
      b(10),
      mu(1.0 / depth_mean),
      z_range(1.0 / depth_min),
      sigma2(z_range * z_range / 36)
{
}

DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector,
                         callback_t seed_converged_cb)
    : feature_detector_(feature_detector),
      seed_converged_cb_(seed_converged_cb),
      seeds_updating_halt_(false),
      thread_(NULL),
      new_keyframe_set_(false),
      new_keyframe_min_depth_(0.0),
      new_keyframe_mean_depth_(0.0)
{
}

DepthFilter::~DepthFilter()
{
  stopThread();
  SVO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread()
{
  thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
  SVO_INFO_STREAM("DepthFilter stop thread invoked.");
  if(thread_ != NULL)
  {
    SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
    seeds_updating_halt_ = true;
    thread_->interrupt();
    thread_->join();
    thread_ = NULL;
  }
}

void DepthFilter::addFrame(FramePtr frame)
{
  if(thread_ != NULL)
  {
    {
      lock_t lock(frame_queue_mut_);
      if(frame_queue_.size() > 2)
        frame_queue_.pop();
      frame_queue_.push(frame);
    }
    seeds_updating_halt_ = false;
    frame_queue_cond_.notify_one();
  }
  else
    updateSeeds(frame);
}

void DepthFilter::addKeyframe(FramePtr frame, double depth_mean,
                              double depth_min)
{
  new_keyframe_min_depth_ = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  if(thread_ != NULL)
  {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;
    seeds_updating_halt_ = true;
    frame_queue_cond_.notify_one();
  }
  else
    initializeSeeds(frame);
}

void DepthFilter::initializeSeeds(FramePtr frame)
{
  Features new_features;
  feature_detector_->setExistingFeatures(frame->fts_);
  feature_detector_->detect(frame.get(), frame->img_pyr_,
                            Config::triangMinCornerScore(), new_features);

  vector<double> depths;
  vector<double> depth_errors;
  SVO_INFO_STREAM("initializeSeeds getStereoDepths");
  getStereoDepths(frame, new_features, depths, depth_errors);

  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
  ++Seed::batch_counter;
  int idx = 0;
  // initialize a seed for every new feature
  for(const auto &ftr : new_features)
  {
    if(depths[idx] == 0)
      seeds_.push_back(
          Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
    else
      seeds_.push_back(Seed(ftr, depths[idx],
                            max(0.01, depths[idx] - 10 * depth_errors[idx])));
    idx++;
  }

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized " << new_features.size()
                                                << " new seeds");
  seeds_updating_halt_ = false;
}

void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  list<Seed>::iterator it = seeds_.begin();
  size_t n_removed = 0;
  while(it != seeds_.end())
  {
    if(it->ftr->frame == frame.get())
    {
      it = seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}

void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false;

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::updateSeedsLoop()
{
  while(!boost::this_thread::interruption_requested())
  {
    FramePtr frame;
    {
      lock_t lock(frame_queue_mut_);
      while(frame_queue_.empty() && !new_keyframe_set_)
        frame_queue_cond_.wait(lock);
      if(new_keyframe_set_)
      {
        new_keyframe_set_ = false;
        seeds_updating_halt_ = false;
        clearFrameQueue();
        frame = new_keyframe_;
      }
      else
      {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }
    updateSeeds(frame);
    if(frame->isKeyframe())
      initializeSeeds(frame);
  }
}

void DepthFilter::updateSeeds(FramePtr frame)
{
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  size_t n_updates = 0, n_failed_matches = 0, n_seeds = seeds_.size();
  lock_t lock(seeds_mut_);

  const double focal_length = frame->cam_->errorMultiplier2();
  double px_noise = 1.0;
  double px_error_angle =
      atan(px_noise / (2.0 * focal_length)) * 2.0; // law of chord (sehnensatz)

  list<Seed>::iterator it = seeds_.begin();

  while(it != seeds_.end())
  {
    // set this value true when seeds updating should be interrupted
    if(seeds_updating_halt_)
      return;

    // check if seed is not already too old
    if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs)
    {
      it = seeds_.erase(it);
      continue;
    }

    // check if point is visible in the current image
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    const Vector3d xyz_f(T_ref_cur.inverse() * (1.0 / it->mu * it->ftr->f));
    if(xyz_f.z() < 0.0)
    {
      ++it; // behind the camera
      continue;
    }
    if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>()))
    {
      ++it; // point does not project in image
      continue;
    }

    // we are using inverse depth coordinates
    float z_inv_min = it->mu + sqrt(it->sigma2);
    float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
    double z;
    if(!matcher_.findEpipolarMatchDirect(*it->ftr->frame, *frame, *it->ftr,
                                         1.0 / it->mu, 1.0 / z_inv_min,
                                         1.0 / z_inv_max, z))
    {
      it->b++; // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }

    // compute tau
    double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
    double tau_inverse =
        0.5 * (1.0 / max(0.0000001, z - tau) - 1.0 / (z + tau));

    // update the estimate
    updateSeed(1. / z, tau_inverse * tau_inverse, &*it);
    ++n_updates;

    if(frame->isKeyframe())
    {
      // The feature detector should not initialize new seeds close to this
      // location
      feature_detector_->setGridOccpuancy(matcher_.px_cur_);
    }

    // if the seed has converged, we initialize a new candidate point and remove
    // the seed
    if(sqrt(it->sigma2) < it->z_range / options_.seed_convergence_sigma2_thresh)
    {
      if(it->ftr->point != NULL)
        SVO_ERROR_STREAM("it->ftr->point != NULL");
      assert(it->ftr->point == NULL); // TODO this should not happen anymore
      // if(it->ftr->point != NULL)
      //  delete it->ftr->point;
      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() *
                         (it->ftr->f * (1.0 / it->mu)));
      Point *point = new Point(xyz_world, it->ftr);
      it->ftr->point = point;
      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new Feature(frame.get(), matcher_.px_cur_,
      matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
      {
        seed_converged_cb_(point, it->sigma2); // put in candidate list
      }
      it = seeds_.erase(it);
    }
    else if(isnan(z_inv_min))
    {
      SVO_WARN_STREAM("z_min is NaN");
      it = seeds_.erase(it);
    }
    else
      ++it;
  }
}

void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr &frame, std::list<Seed> &seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it = seeds_.begin(); it != seeds_.end(); ++it)
  {
    if(it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

void DepthFilter::updateSeed(const float x, const float tau2, Seed *seed)
{
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if(std::isnan(norm_scale))
    return;
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  float s2 = 1.0f / (1.0f / seed->sigma2 + 1.0f / tau2);
  float m = s2 * (seed->mu / seed->sigma2 + x / tau2);
  float C1 = seed->a / (seed->a + seed->b) * boost::math::pdf(nd, x);
  float C2 = seed->b / (seed->a + seed->b) * 1.0f / seed->z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f = C1 * (seed->a + 1.0f) / (seed->a + seed->b + 1.0f) +
            C2 * seed->a / (seed->a + seed->b + 1.0f);
  float e = C1 * (seed->a + 1.0f) * (seed->a + 2.0f) /
                ((seed->a + seed->b + 1.0f) * (seed->a + seed->b + 2.0f)) +
            C2 * seed->a * (seed->a + 1.0f) /
                ((seed->a + seed->b + 1.0f) * (seed->a + seed->b + 2.0f));

  // update parameters
  float mu_new = C1 * m + C2 * seed->mu;
  seed->sigma2 = C1 * (s2 + m * m) + C2 * (seed->sigma2 + seed->mu * seed->mu) -
                 mu_new * mu_new;
  seed->mu = mu_new;
  seed->a = (e - f) / (f - e / f);
  seed->b = seed->a * (1.0f - f) / f;
}

double DepthFilter::computeTau(const SE3 &T_ref_cur, const Vector3d &f,
                               const double z, const double px_error_angle)
{
  Vector3d t(T_ref_cur.translation());
  Vector3d a = f * z - t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f.dot(t) / t_norm); // dot product
  double beta = acos(a.dot(-t) / (t_norm * a_norm)); // dot product
  double beta_plus = beta + px_error_angle;
  double gamma_plus = PI - alpha - beta_plus; // triangle angles sum to PI
  double z_plus = t_norm * sin(beta_plus) / sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}

bool DepthFilter::getStereoDepths(const FramePtr &frame,
                                  const Features &features,
                                  vector<double> &depths,
                                  vector<double> &depth_errors)
{
  vector<cv::Point2f> px_left;
  vector<int> level_0_ftr_indices;
  px_left.reserve(features.size());
  level_0_ftr_indices.reserve(features.size());
  int idx = 0;
  for(const auto &ftr : features)
  {
    if(ftr->level == 0)
    {
      px_left.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
      level_0_ftr_indices.push_back(idx);
    }
    idx++;
  }

  SVO_INFO_STREAM(
      "getStereoDepths: Num 0th level features: " << px_left.size());
  if(px_left.size() == 0)
  {
    depths.clear();
    depths.resize(features.size());
    return false;
  }

  vector<cv::Point2f> px_right;
  const int klt_win_size = 30;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                            klt_max_iter, klt_eps);
  cv::calcOpticalFlowPyrLK(frame->img(), frame->img_right(), px_left, px_right,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size), 2, termcrit);

  auto px_left_it = px_left.begin();
  auto px_right_it = px_right.begin();
  vector<double> disparities;
  depths.clear();
  depth_errors.clear();
  depths.resize(features.size());
  depth_errors.resize(features.size());

  for(size_t i = 0; px_left_it != px_left.end(); ++i)
  {
    double dx = px_left_it->x - px_right_it->x;
    double dy = px_left_it->y - px_right_it->y;

    if(!status[i] || dy > 3)
    {
      px_left_it = px_left.erase(px_left_it);
      px_right_it = px_right.erase(px_right_it);
      continue;
    }
    disparities.push_back(dx);
    const double depth = frame->baseline_px_ / dx;
    depths[level_0_ftr_indices[i]] = depth;
    depth_errors[level_0_ftr_indices[i]] =
        depth / dx; // Assuming 1 px error in disparity
    ++px_left_it;
    ++px_right_it;
  }

  SVO_INFO_STREAM("getStereoDepths: KLT tracked " << disparities.size()
                                                  << " features");

  return true;
}

} // namespace svo
