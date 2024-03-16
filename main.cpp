#include<csyslib>
#include<cmath>
#include<matplotlibcpp.h>
#include<Eigen>
#include<winuser.h>
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
	rs::config c;
	c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200);
	c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200);

	p.start(c);


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

	float dt = 0.1 //time-step
	
	bool show_animation = true;
	
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

int main()
{
	float time = 0.0;	

	//state vector 
	MatrixXd::Zero xEst(4,1);
	MatrixXd::Zero xTrue(4,1);
	MatirxXd::Identity PESt(4,1);

	//history 
	stack<MatrixXd> hxEst;
	stack<MtrixXd> hxTrue;

	hxEst = stack.push(xEst);
	hxTrue = stack.push(xTrue);

	while True:
	{
		//IMU data from realsense
		frames = p.wait_for_frames()

		raw_accel = frames[0].as_motion_frames().get_motion_data();
		raw_gyro = frames[1].as_motion_frames().get_motion_data();

		MatrixXd rs_to_base_tfm(3,3);
		rs_to_base_tfm<< 0, 0, 1,
				 1, 0, 0,
			 	 0, 0, 1;
       
		MatrixXd accel(1,3);
		accel<< raw_accel.x,raw_accel.y, raw_accel.z;
		accel = accel * rs_to_base_tfm;
		accel = accel.transpose();

		MatrixXd gyro(1,3);
		gyro<< raw_gyro.x, raw_gyro.y, raw_gyro.z;
		gyro = gyro * rs_to_base_tfm;
		gyro = gyro.transpose();

		//calculating net acceleration
		accel_net = sqrt((pow(accel(0), 2) + pow(accel(1), 2)));

		MatrixXd u(2,1);
		u<< accel_net*dt, gyro(2); //control input
				
		time+ = dt;

		tie(xTrue, ud) = observation(xTrue, u);

		z = observation_model(xTrue);

		tie(xEst, PEst) = ekf_estimation(xEst, PEst,x ,ud);

		//store datat history
		hxEst = stack.push(xEst);
		hxTrue = stack.push(xTrue);

		if show_animation
		{
			plt.cla();

			//for stopping simulation with the esc key
			plt.gcf().canvas.mpl_connect("key release event", 
				[]{if(GetKeyState((int)"q"==1)) exit(0); 
				   else continue;});	

			//plotting actual state(represented by blue)
			plt.plot(hxTrue.coeff(0, seq(0, hxTrue.cols()), 
				 hxTrue.coeff(1, seq(1, hxTrue.cols()), "bo-");


			//plotting actual state(represented y red)
			plt.plot(hxEst.coeff(0, seq(0, hxEst.cols()), 
				 hxEst.coeff(1, seq(1, hxEst.cols()), "r-");

			plt.axis("equal");

			plt.grid(true);

			plt.pause(0.001);
		}
	}
}




	     	











