#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);
  Q_base_  = MatrixXd(4, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0     ,
              0     , 0.0225
  ;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0     , 0   ,
              0   , 0.0009, 0   ,
              0   , 0     , 0.09
  ;
  
  // Init kalman filter
  ekf_.x_ = VectorXd(4);
  ekf_.F_ = MatrixXd::Identity(4, 4);
  ekf_.P_ = MatrixXd::Identity(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.H_ = MatrixXd(2, 4);
  ekf_.H_ << 1,0,0,0,
             0,1,0,0
  ;
  
  // Init Q_base so we won't need to recompute all the time
  // we will only need to multiply by the right deltat power
  // Use noise_ax = 9 and noise_ay = 9 as per project requirement
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  Q_base_ << noise_ax/4.0, 0.0         , noise_ax/2.0, 0.0         ,
             0.0         , noise_ay/4.0, 0.0         , noise_ay/2.0,
             noise_ax/2.0, 0.0         , noise_ax    , 0.0         ,
             0.0         , noise_ay/2.0, 0.0         , noise_ay
  ;

  // Use the noise to init the covariance matrix P
  ekf_.P_ *= noise_ax;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Init state by converting polar to cartensian coordinates
      // As well as the velocity
      float ro     = measurement_pack.raw_measurements_(0);
      float phi    = measurement_pack.raw_measurements_(1);
      float ro_dot = measurement_pack.raw_measurements_(2);
      
      ekf_.x_ <<  ro     * cos(phi), ro     * sin(phi),
                  ro_dot * cos(phi), ro_dot * sin(phi)
      ;  
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);

      ekf_.x_ <<  px , py ,
                  0.0, 0.0
      ;  
    }

    // init timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  // Compute elapsed time since last measurement
  float deltat = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Update state transition matrix
  ekf_.F_(0, 2) = deltat;
  ekf_.F_(1, 3) = deltat;

  // Update the process noise matrix
  ekf_.Q_ = Q_base_;
  
  float deltat2 = deltat  * deltat;
  float deltat3 = deltat2 * deltat;
  float deltat4 = deltat3 * deltat;
  
  ekf_.Q_(0, 0) *= deltat4;
  ekf_.Q_(0, 2) *= deltat3;
  ekf_.Q_(1, 1) *= deltat4;
  ekf_.Q_(1, 3) *= deltat3;
  ekf_.Q_(2, 0) *= deltat3;
  ekf_.Q_(2, 2) *= deltat2;
  ekf_.Q_(3, 1) *= deltat3;
  ekf_.Q_(3, 3) *= deltat2;

  // Now we can predict
  ekf_.Predict();
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

#ifdef __DEBUG__
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
#endif
}
