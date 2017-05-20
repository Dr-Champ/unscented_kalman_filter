#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <math.h>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2; 

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO: Done !!

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  zero_div_threshold = 0.001;
  time_us_ = 0;
  Xsig_pred_ = MatrixXd(n_x_, (2 * n_aug_) + 1);
  lambda_ = 3 - n_aug_;
  NIS_radar_ = 0;
  NIS_laser_ = 0;

  // This is only needed if we use linear Kalman filter with laser
  /*H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;*/

  R_radar = MatrixXd(3, 3);
  R_radar << std_radr_ * std_radr_, 0, 0,
             0, std_radphi_ * std_radphi_, 0,
             0, 0, std_radrd_ * std_radrd_;
  R_laser = MatrixXd(2, 2);
  R_laser << std_laspx_ * std_laspx_, 0,
             0, std_laspy_ * std_laspy_;

  // set weights for combining sigma vectors to new predicted state
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (n_aug_ + lambda_);

  weights_ = VectorXd((2 * n_aug_) + 1);
  weights_p_ = VectorXd((2 * n_aug_) + 1);
  weights_(0) = weight_0;
  weights_p_(0) = weight_0;

  // according to wiki https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
  // the weight 0 for computing covariance matrix should be larger by approx 3. But this 
  // doesn't seem to improve the RMSE
  //weights_p_(0) = weight_0 + 3;

  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    weights_(i) = weight;
    weights_p_(i) = weight;
  }
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO: Done !!

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    double px = 0;
    double py = 0;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_(0);
      py = meas_package.raw_measurements_(1);
    } 
    else {
      double ro = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      px = ro * cos(phi);
      py = ro * sin(phi);
    }
    x_ << px, py, 0.5, 0.95, 0;

    P_ << 1.0, 0, 0, 0, 0,
          0, 1.0, 0, 0, 0,
          0, 0, 1.0, 0, 0,
          0, 0, 0, 1.0, 0,
          0, 0, 0, 0, 1.0;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // sec

  if (((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) || 
      ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_))) {
    try {
      Prediction(delta_t);
    }
    catch (std::range_error e) {
      // reset the P_ to ensure being positive definite
      P_ << 1.0, 0, 0, 0, 0,
            0, 1.0, 0, 0, 0,
            0, 0, 1.0, 0, 0,
            0, 0, 0, 1.0, 0,
            0, 0, 0, 0, 1.0;
      Prediction(delta_t);
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    } 
    else {
      UpdateLidar(meas_package);
    }
    time_us_ = meas_package.timestamp_;
  } 
  else {
    cout << "Ignoring sensor input" << endl;
  }

  cout << "Updated x " << endl << x_ << endl;
  cout << "Updated P " << endl << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO: Done !!

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  // generate (augmented) sigma points from augmented P_a
  MatrixXd P_a = MatrixXd(n_aug_, n_aug_);
  P_a.fill(0.0);
  P_a.topLeftCorner(n_x_, n_x_) = P_;
  P_a(n_x_, n_x_) = std_a_ * std_a_;
  P_a(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  Eigen::LLT<MatrixXd> lltOfPaug(P_a);
  if (lltOfPaug.info() == Eigen::NumericalIssue) {
    // Based on https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449
    // At this point no state variables have been modified yet. Just throw exception and reset the 
    // P_ on the outside
    cout << "LLT failed!" << endl;
    throw std::range_error("LLT failed");
  }

  MatrixXd L = lltOfPaug.matrixL();
  VectorXd x_a = VectorXd(n_aug_);
  x_a.head(n_x_) = x_;
  x_a(n_x_) = 0;
  x_a(n_x_ + 1) = 0;

  MatrixXd Xsig_a_ = MatrixXd(n_aug_, (2 * n_aug_) + 1);
  Xsig_a_.col(0) = x_a;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_a_.col(i + 1) = x_a + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_a_.col(i + 1 + n_aug_) = x_a - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // map sigma points to the new state space
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    Xsig_pred_.col(i) = computePrediction(Xsig_a_.col(i), delta_t); 
  }

  // compute predicted mean 
  x_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  x_(3) = normaliseAngle(x_(3));

  // compute predicted P 
  P_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normaliseAngle(x_diff(3));
    P_ += weights_p_(i) * x_diff * x_diff.transpose();
  }  

  cout << "Predicted x " << endl << x_ << endl;
  cout << "Predicted P " << endl << P_ << endl;
}

VectorXd UKF::computePrediction(const VectorXd &x_a, double delta_t) {
  double px = x_a(0);
  double py = x_a(1);
  double v = x_a(2);
  double yaw = x_a(3);
  double yawd = x_a(4);
  double nu_a = x_a(5);
  double nu_yawdd = x_a(6);

  VectorXd x_pred = VectorXd(n_x_);
  x_pred << px, py, v, yaw, yawd;

  // add incremental part
  if (fabs(yawd) < zero_div_threshold) {
    x_pred(0) += v * cos(yaw);
    x_pred(1) += v * sin(yaw);
  }
  else {
    double yaw_pred = yaw + yawd * delta_t;
    x_pred(0) += (v / yawd) * (sin(yaw_pred) - sin(yaw));
    x_pred(1) += (v / yawd) * (cos(yaw) - cos(yaw_pred));
  }

  x_pred(3) += yawd * delta_t;

  // add the noise part
  double temp_t = 0.5 * delta_t * delta_t;
  x_pred(0) += temp_t * nu_a * cos(yaw);
  x_pred(1) += temp_t * nu_a * sin(yaw);
  x_pred(2) += nu_a * delta_t;
  x_pred(3) += temp_t * nu_yawdd;
  x_pred(4) += nu_yawdd * delta_t;

  x_pred(3) = normaliseAngle(x_pred(3));

  return x_pred;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO: Done !!

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_laser = 2;

  // Use linear Kalman filter for laser .. doesn't make a difference though
  /*double px = meas_package.raw_measurements_(0);
  double py = meas_package.raw_measurements_(1);
  VectorXd z = VectorXd(n_laser);
  z << px, py;

  VectorXd z_innovation = z - (H_laser_ * x_);
  MatrixXd S = (H_laser_ * P_ * H_laser_.transpose()) + R_laser;
  MatrixXd K = P_ * H_laser_.transpose() * S.inverse();
  x_ = x_ + (K * z_innovation);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;*/

  // Use Unscented Kalman filter for layer
  // project sigma points from state to measurement space
  MatrixXd Z_sig = MatrixXd(n_laser, (2 * n_aug_) + 1);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    VectorXd current_xsig_pred = Xsig_pred_.col(i);
    Z_sig.col(i) << current_xsig_pred(0), current_xsig_pred(1);
  }  

  // compute mean of the projected measurement points
  VectorXd z_mean = VectorXd(n_laser);
  z_mean.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    z_mean += weights_(i) * Z_sig.col(i);
  }

  // compute covariance matrix S and cross-correlation matrix T
  MatrixXd S = MatrixXd(n_laser, n_laser);
  MatrixXd T = MatrixXd(n_x_, n_laser);
  S.fill(0.0);
  T.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    VectorXd z_diff = Z_sig.col(i) - z_mean;
    S += weights_p_(i) * z_diff * z_diff.transpose();

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    T += weights_p_(i) * x_diff * z_diff.transpose();
  }
  S += R_laser;

  // compute Kalman gain K
  MatrixXd K = T * S.inverse();

  // Update state x_
  double px = meas_package.raw_measurements_(0);
  double py = meas_package.raw_measurements_(1);
  VectorXd z = VectorXd(n_laser);
  z << px, py;
  VectorXd z_innovation = z - z_mean;
  x_ += K * z_innovation;
  x_(3) = normaliseAngle(x_(3));

  // Update covariance matrix P_
  P_ -= K * S * K.transpose();

  // Compute NIS
  NIS_laser_ = z_innovation.transpose() * S.inverse() * z_innovation;

  cout << "updating from lidar " << endl 
      << "weights_ " << weights_ << endl
      << "Xsig_pred_: " << Xsig_pred_ << endl
      << "Z_sig " << Z_sig << endl
      << "z: " << z << endl 
      << "z_innovation: "<< z_innovation << endl 
      << "T: " << T << endl
      << "S: " << S << endl
      << "K: " << K << endl
      << "NIS: " << NIS_laser_ << endl; 
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO: Done !!

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  // project sigma points from state to measurement space
  int n_radar = 3;
  MatrixXd Z_sig = MatrixXd(n_radar, (2 * n_aug_) + 1);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    VectorXd z_sig = projectStateToRadarSpace(Xsig_pred_.col(i));
    z_sig(1) = normaliseAngle(z_sig(1));
    Z_sig.col(i) = z_sig;
  }  

  // compute mean of the projected measurement points
  VectorXd z_mean = VectorXd(n_radar);
  z_mean.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    z_mean += weights_(i) * Z_sig.col(i);
  }
  z_mean(1) = normaliseAngle(z_mean(1));

  // compute covariance matrix S and cross-correlation matrix T
  MatrixXd S = MatrixXd(n_radar, n_radar);
  MatrixXd T = MatrixXd(n_x_, n_radar);
  S.fill(0.0);
  T.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {
    VectorXd z_diff = Z_sig.col(i) - z_mean;
    z_diff(1) = normaliseAngle(z_diff(1));
    S += weights_p_(i) * z_diff * z_diff.transpose();

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normaliseAngle(x_diff(3));
    T += weights_p_(i) * x_diff * z_diff.transpose();
  }
  S += R_radar;

  // compute Kalman gain K
  MatrixXd K = T * S.inverse();

  // Update state x_
  double ro = meas_package.raw_measurements_(0);
  double phi = meas_package.raw_measurements_(1);
  double ro_dot = meas_package.raw_measurements_(2);
  VectorXd z = VectorXd(n_radar);
  z << ro, phi, ro_dot;
  VectorXd z_innovation = z - z_mean;
  x_ += K * z_innovation;
  x_(3) = normaliseAngle(x_(3));

  // Update covariance matrix P_
  P_ -= K * S * K.transpose();

  // Compute NIS
  NIS_radar_ = z_innovation.transpose() * S.inverse() * z_innovation;

  cout << "updating from radar " << endl 
      << "weights_ " << weights_ << endl
      << "Xsig_pred_: " << Xsig_pred_ << endl
      << "Z_sig " << Z_sig << endl
      << "z: " << z << endl 
      << "z_innovation: "<< z_innovation << endl 
      << "T: " << T << endl
      << "S: " << S << endl
      << "K: " << K << endl
      << "NIS: " << NIS_radar_ << endl; 
}

VectorXd UKF::projectStateToRadarSpace(const VectorXd &x) {
  double px = x(0);
  double py = x(1);
  double v = x(2);
  double yaw = x(3);
  double vx = v * cos(yaw);
  double vy = v * sin(yaw);

  VectorXd z = VectorXd(3);
  double c1 = sqrt(px * px + py * py);
  double px_non_zero = px;

  /*if (c1 < 0.001) {
    c1 = 0.001;
  }*/

  if ((px_non_zero < zero_div_threshold) && (px_non_zero > 0)) {
    px_non_zero = zero_div_threshold;
  }
  else if ((px_non_zero > -zero_div_threshold) && (px_non_zero <= 0)) {
    px_non_zero = -zero_div_threshold;
  }

  z << c1,
       atan2(py, px_non_zero),
       (px * vx + py * vy) / c1;
  return z;
}

double UKF::normaliseAngle(double& zeta) {
  while (zeta > M_PI) {zeta -= 2 * M_PI;}
  while (zeta < -M_PI) {zeta += 2 * M_PI;}
  return zeta;
}












