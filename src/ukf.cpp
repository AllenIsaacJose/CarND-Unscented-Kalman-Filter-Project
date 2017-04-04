#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
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

  // state dimension
  n_x_ = 5;

  // augmented state dimension
  n_aug_ = 7;

  // sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

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
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);;

  // weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

  if (dt > 0.001) {
    Prediction(dt);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  // generate sigma points
  Xsig.col(0) = x_;
  for (int i=0; i< n_x_; i++) {
    Xsig.col(i+1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  Q(0, 0) = std_a_*std_a_;
  Q(1, 1) = std_yawdd_*std_yawdd_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i< n_aug_; i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * A_aug.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A_aug.col(i);
  }

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for (int i=0; i< 2 * n_aug_ + 1; i++) {
    double px = Xsig_aug.col(i)(0);
    double py = Xsig_aug.col(i)(1);
    double v = Xsig_aug.col(i)(2);
    double mu = Xsig_aug.col(i)(3);
    double mu_dot = Xsig_aug.col(i)(4);
    double a = Xsig_aug.col(i)(5);
    double yawdd = Xsig_aug.col(i)(6);

    VectorXd x = VectorXd(n_x_);
    x << px, py, v, mu, mu_dot;

    if (fabs(mu_dot) > 0.0001) {
        VectorXd A = VectorXd(n_x_);
        A << (v/mu_dot)*(sin(mu + mu_dot*delta_t) - sin(mu)),
             (v/mu_dot)*(-cos(mu + mu_dot*delta_t) + cos(mu)),
             0,
             mu_dot*delta_t,
             0;

        VectorXd B = VectorXd(n_x_);
        B << 0.5*delta_t*delta_t*cos(mu)*a,
             0.5*delta_t*delta_t*sin(mu)*a,
             delta_t*a,
             0.5*delta_t*delta_t*yawdd,
             delta_t*yawdd;

        Xsig_pred_.col(i) = x + A + B;
    } else {
        VectorXd A = VectorXd(n_x_);
        A << v*cos(mu)*delta_t,
             v*sin(mu)*delta_t,
             0,
             mu_dot*delta_t,
             0;

        VectorXd B = VectorXd(n_x_);
        B << 0.5*delta_t*delta_t*cos(mu)*a,
             0.5*delta_t*delta_t*sin(mu)*a,
             delta_t*a,
             0.5*delta_t*delta_t*yawdd,
             delta_t*yawdd;

        Xsig_pred_.col(i) = x + A + B;
    }
  }

  int n_sigma = 2 * n_aug_ + 1;

  //set weights
  weights_.fill(0.0);
  for (int i=0; i < n_sigma; i++) {
      if (i == 0) {
        weights_(i) = lambda_ / (lambda_ + n_aug_);
      } else {
        weights_(i) = 0.5 * (1 / (lambda_ + n_aug_));
      }
  }

  //create vector for predicted state
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);

  //create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);

  //predict state mean
  for (int i=0; i < n_sigma; i++) {
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  for (int i=0; i < n_sigma; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;

    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    P_pred +=  weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
