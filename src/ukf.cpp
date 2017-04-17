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
  Xsig_pred_ = MatrixXd(2*n_aug_+1, n_x_);

  // weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(1/(2*(lambda_+n_aug_)));
  weights_(0) = lambda_/(lambda_+n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MatrixXd} Xsig_out The generated sigma points matrix.
 */
void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //set example state
  VectorXd x = x_;

  //set example covariance matrix
  MatrixXd P = P_;

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P.llt().matrixL();

  //set first column of sigma point matrix
  Xsig.col(0) = x;

  //set remaining sigma points
  for (int i=0; i< n_x_; i++) {
    Xsig.col(i+1) = x + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x - sqrt(lambda_ + n_x_) * A.col(i);
  }

  //print result
  //std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  //write result
  *Xsig_out = Xsig;

}

/**
 * @param {MatrixXd} Xsig_out The sigma points matrix to be augmented.
 */
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //set example state
  VectorXd x = x_;

  //create example covariance matrix
  MatrixXd P = P_;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P;
  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  Q(0, 0) = std_a_ * std_a_;
  Q(1, 1) = std_yawdd_ * std_yawdd_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i< n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  //print result
  // std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;

}

/**
 * @param {MatrixXd} Xsig_aug The augmented sigma points to be precessed.
 * @param {MatrixXd} Xsig_out The predicted sigma points.
 * @param {double} delta_t The time elapsed since last measurement.
 */
void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, MatrixXd* Xsig_out, double delta_t) {

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for (int i=0; i< 2 * n_aug_ + 1; i++) {
    double px = Xsig_aug.col(i)(0);
    double py = Xsig_aug.col(i)(1);
    double v = Xsig_aug.col(i)(2);
    double psi = Xsig_aug.col(i)(3);
    double psi_dot = Xsig_aug.col(i)(4);
    double a = Xsig_aug.col(i)(5);
    double yawdd = Xsig_aug.col(i)(6);

    VectorXd x = VectorXd(n_x_);
    x << px, py, v, psi, psi_dot;

    if (fabs(psi_dot) > 0.0001) {
        VectorXd A = VectorXd(n_x_);
        A << (v/psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi)),
             (v/psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi)),
             0,
             psi_dot * delta_t,
             0;

        VectorXd B = VectorXd(n_x_);
        B << 0.5*delta_t*delta_t*cos(psi)*a,
             0.5*delta_t*delta_t*sin(psi)*a,
             delta_t*a,
             0.5*delta_t*delta_t*yawdd,
             delta_t*yawdd;

        Xsig_pred.col(i) = x + A + B;
    } else {
        VectorXd A = VectorXd(n_x_);
        A << v*cos(psi)*delta_t,
             v*sin(psi)*delta_t,
             0,
             psi_dot*delta_t,
             0;

        VectorXd B = VectorXd(n_x_);
        B << 0.5*delta_t*delta_t*cos(psi)*a,
             0.5*delta_t*delta_t*sin(psi)*a,
             delta_t*a,
             0.5*delta_t*delta_t*yawdd,
             delta_t*yawdd;

        Xsig_pred.col(i) = x + A + B;
    }
  }

  //print result
  // std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;
}

/**
 * @param {VectorXd} x_out The predicted state vector.
 * @param {MatrixXd} P_out The predicted covariance matrix.
 */
void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predict state mean
  x.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    x += weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P +=  weights_(i) * x_diff * x_diff.transpose();
  }

  //print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P << std::endl;

  //write result
  *x_out = x;
  *P_out = P;
}

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
    time_us_ = meas_package.timestamp_;
    x_ << 1, 1, 0, 0, 0;
    P_ << 1, 0, 0, 0, 0,
  			  0, 1, 0, 0, 0,
		      0, 0, 1, 0, 0,
		      0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float range = meas_package.raw_measurements_[0];
      float angle = meas_package.raw_measurements_[1];
      float range_rate = meas_package.raw_measurements_[2];

      float px = range * cos(angle);
      float py = range * sin(angle);
      float vx = range_rate * cos(angle);
      float vy = range_rate * sin(angle);

      x_ << px, py, 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];

      if (px == 0 || py == 0){
        cout << "Ignoring empty laser measurement" << endl;
        return;
      }

      x_ << px, py, 0, 0, 0;
    }

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

  //generate sigma points
  MatrixXd Xsig = MatrixXd(2 * n_x_ + 1, n_x_);
  GenerateSigmaPoints(&Xsig);

  //augment sigma points
  MatrixXd Xsig_aug = MatrixXd(2 * n_aug_ + 1, n_aug_);
  AugmentedSigmaPoints(&Xsig_aug);

  //predict sigma points
  SigmaPointPrediction(Xsig_aug, &Xsig_pred_, delta_t);

  //create vector for predicted state
  VectorXd x_pred = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);

  //predict state mean and state covariance matrix
  PredictMeanAndCovariance(&x_pred, &P_pred);

  x_ = x_pred;
  P_ = P_pred;
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

  // VectorXd z = meas_package.raw_measurements_;
  //
  // //create matrix for cross correlation Tc
  // MatrixXd Tc = MatrixXd(n_x_, 3);
  //
  // //calculate cross correlation matrix
  // Tc.fill(0.0);
  // for (int i=0; i < 2 * n_aug_ + 1; i++) {
  //   Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  // }
  //
  // //calculate Kalman gain K;
  // MatrixXd K = Tc * S.inverse();
  //
  // //update state mean and covariance matrix
  // x_ = x_ + K * (z - z_pred);
  // P_ = P_ - K * S * K.transpose();

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
