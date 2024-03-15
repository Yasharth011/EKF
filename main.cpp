#include<iostream>
#include<csyslib>
#include<cmath>
#include<matplotlibcpp.h>
#include<Eigen/Dense>
#include<EigenRand/EigenRand>
#include<libealsense2/rs.hpp>
#include<stack>
#include<tuple>

namespace plt = matplotlibcpp.h;
namespace rs = librealsense2/rs.hpp;
using namespace std;
using namespace Eigen;

class EKF
{
	//pipeline for realsense
	rs::pipeline p;

	p.start();


	//covariance matrix
	MatrixXd Q(3,3);
	Q<< .1, 0, 0,
    	0, .1, 0,
    	0, 0, .1;

	MatrixXd R(2,2);
	R<< 1,0,
    	    0,1;

	//noise parameter
	ip_noise = MatrixXd(2,2);
	ip_noise<< 1, 0,
	           0, (30*(3.14/180));

	//measurement matrix
	MatrixXd H(2,4);
	H<< 1,0,0,0,
	    0,1,0,0;

	dt = 0.1 //time-step
	
	tuple<MatrixXd, MatrixXd> MatrixXf observation(MatrixXd xTrue(4,1), MatrixXd u(2,1))
	{
		xTrue = state_model(xTrue, u);

		//adding noise to the input
		MatrixXd ud(2,1);
		ud = u = ip_noise * MatrixXd::random(2,1);

		return make_tuple(xTrue, ud);
	}

	MatirxXf state_model(MatrixXd x(4,1), MatrixXd u(2,1))
	{
		MatrixXd A(4,4);
		A<< 1,0,0,0,
		    0,1,0,0,
		    0,0,1,0,
		    0,0,0,0;

		MatrixXd B(4,2);
		B<< (dt*cos(x.coeff(2,0), 0,
		    (dt*sin(x.coeff(2,0), 0,
		    0, dt,
		    1, 0;

		x = A * x + B * u;

		return x;
	}

	MatrixXf jacob_f(MatrixXd x(3,1), MatrixXd u(2,1))
	{
		yaw = x.coeff(2,0);

		v = u.coeff(0,0);

		MatrixXd jF(4,4);

		jF<< 1.0, 0.0, (-dt*v*sin(yaw)), (dt*cos(yaw)),
		     0.0, 1.0, (dt*v*cos(yaw)), (dt*sin(yaw)),
		     0.0, 0.0, 1.0, 0.0,
		     0.0, 0.0, 0.0, 1.0;
		
		return jF;
	}

	MatrixXf observation_model(MatrixXd x)
	{
		MatrixXd z(4,1);

		z = H * x;

		retrun z;
	}
   
	tuple<MatrixXd, MatrixXd> MatrixXf efk_estimation(MatrixXd xEst(4,1), MatrixXd PEst(4,4) ,  MatrixXd z(4,1), MatrixXd u(2,1))
	{
		//Predict 
		MatrixXd xPred(4,1);
		xPred = state_model(xEst, u);

		//state vector 
		jF = jacob_f(xEst, u);
		PPred = jF*PEst*jF + Q;

		//Update
		MatrixXd zPred(4,1);
		zPred = observation_model(xPred);

		MatrixXd y(4,1);
		y = z - zPred; //measurement residual 
		
		S = H*PPred*H.transpose() + R; //Innovation Covariance
		
		K = PPred*H.tranpose()*S.inverse(); //Kalman Gain
		
		xEst = xPred + K * y; //update step

		PEst = (MatrixXd::Identity(4,1) - (K*H)) * PPred;

		return make_tuple(xEst, Pest);
	}

			


	     	











