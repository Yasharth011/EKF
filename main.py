import sys
import math
import numpy as np
import plot
import matplotlib.pyplot as plt

import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
pipeline.start(config)

#covariance matrix
Q = np.diag([1, #var(x)
             1, #var(y)
             np.deg2rad(1.0), #var(yaw)
             ])**2
R = np.diag([1,1,1])**2

#noise parameter
input_noise = np.diag([1.0,np.deg2rad(5)])**2

#measurement matrix
H = np.diag([1,1,1])**2

dt = 0.1 # time-step

show_animation = True


def observation(xTrue, u):
    xTrue = state_model(xTrue, u)

    #adding noise to input
    ud = u + input_noise @ np.random.randn(2,1)

    return xTrue, ud

def state_model(x, u):

   A = np.diag([1,1,1])**2

   B = np.array([[dt * math.cos(x[2,0]), 0],
                 [dt * math.sin(x[2,0]), 0],
                 [0, dt]])
    
   x = A @ x + B @ u

   return x

def observation_model(x):

    z = H @ x 

    return z

def ekf_estimation(xEst, PEst, z, u):
    
    #Predict 
    xPred = state_model(xEst, u)
    #state covariance
    F = np.diag([1,1,1])**2
    PPred = F*PEst*F.T + Q

    #Update
    zPred = observation_model(xPred)

    y = z - zPred #measurement residual
    
    S = H @ PPred @ H.T + R #Innovation covariance

    K = PPred @ H.T @ np.linalg.inv(S) #kalman gain

    xEst = xPred + K @ y #updating state

    PEst = ((np.eye(3)) - K@H) @ PPred

    return xEst, PEst

def main():

    time = 0.0

    #state vector 
    xEst = np.zeros((3,1))
    xTrue = np.zeros((3,1))
    PEst = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
    
    #history
    hxEst = xEst
    hxTrue = xTrue 

    while True:

        frames = pipeline.wait_for_frames()

        raw_accel = frames[0].as_motion_frame().get_motion_data()
        raw_gyro = frames[1].as_motion_frame().get_motion_data()

        accel = np.asarray([raw_accel.x, raw_accel.y, raw_accel.z])
        gyro = np.asarray([raw_gyro.x, raw_gyro.y, raw_gyro.z])
        
        #calculating net acceleration
        accel_net = math.sqrt((pow(accel[0],2) + pow(accel[2],2)))

        u = np.array([[accel_net*dt], [gyro[0]]]) #control input
        
        time+= dt

        xTrue, ud = observation(xTrue, u)

        z = observation_model(xTrue)

        xEst, Pest = ekf_estimation(xEst, PEst, z, ud)

        #store data histroy 
        hxEst = np.hstack((hxEst, xEst))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()

            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            #plotting actual state (represented by blue line)
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
            
            #plotting estimated state (represented by red line)
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r")


            plot.plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
            
if __name__ == '__main__':
    main()
