#include <math.h>
#include "tools.h"
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
  I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  // State prediction
  x_ = F_ * x_;
  
  // Covariance prediction
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Error vector
  VectorXd y = z - H_ * x_;
  
  // Common update
  return UpdateCommon(y, H_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //Get state parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  //check division by zero
  if (px == 0.0) {
      //throw "Division by zero";
      return;
  }
  
  float norm = sqrt( px*px + py*py );
  
  // Compute the measurement function
  // This can't be our H matrix as the operation is non linear
  // We use directly the transformation from cartesian to polar.
  VectorXd h(3);
  h << norm, atan2(py,px), (px*vx + py*vy) / norm;
  
  // Compute the jacobian measurement matrix
  Tools tools;
  MatrixXd Hj = tools.CalculateJacobian(x_);
  
  // Error vector
  VectorXd y = z - h;
  
  // Common update with the jacobian matrix
  return UpdateCommon(y, Hj);
}


void KalmanFilter::UpdateCommon(const VectorXd &y, const MatrixXd &H) {
  MatrixXd PHt = P_ * H.transpose();
  
  MatrixXd S = H * PHt + R_;
  MatrixXd K = PHt * S.inverse();
  
  // State update
  x_ = x_ + K * y;
  
  // Covariance update
  P_ = (I_ - K * H) * P_; 
}
