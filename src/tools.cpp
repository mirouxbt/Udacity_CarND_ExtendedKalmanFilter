#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0.0, 0.0, 0.0, 0.0;

  // Validation
  if(    estimations[0].size() != 4 
      || estimations.size() != ground_truth.size() ) {
    std::cerr <<  "CalculateRMSE : Invalid or mismatch size of parameters" << std::endl;
    return rmse;
  }
  
  // accumulate diff
  for(size_t i=0; i<estimations.size(); ++i) {
    VectorXd acc = estimations[i] - ground_truth[i];
    acc = acc.array() * acc.array();
    
    rmse += acc;
  }
  
  // Average it
  rmse /= estimations.size();
  
  // Square root it
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);

  //Get state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  if (px == 0.0 && py == 0.0) {
    std::cerr << "CalculateJacobian: Division by zero" << std::endl;
    Hj << 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0
    ;
    return Hj;
  }
  
  //compute the Jacobian matrix
  float norm_square = px*px + py*py;
  float norm        = sqrt(norm_square);
  float norm_32     = norm_square * norm;
  float pv          = px*vy - py*vx;

  Hj <<  px    / norm         , py    / norm         , 0.0        , 0.0,
        -py    / norm_square  , px    / norm_square  , 0.0        , 0.0,
        -py*pv / norm_32      , px*pv / norm_32      , px  / norm , py / norm
  ;
    
  return Hj;
}
