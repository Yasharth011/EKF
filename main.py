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

Q = np.diag([.1, #var(x)
              .1, #var(y)
              np.deg2rad(1), #var(yaw)
              1  #var(velocity)
              ])**2

R = np.diag([1,1])**2

#noise parameter
input_noise = np.diag([1.0,np.deg2rad(30)])**2

#measurement matrix
H = np.array([[1,0,0,0],
              [0,1,0,0]])

dt = 0.1 # time-step

show_animation = True


def observation(xTrue, u):
    xTrue = state_model(xTrue, u)

    #adding noise to input
    ud = u + input_noise @ np.random.randn(2,1)

    return xTrue, ud

def state_model(x, u):

   A = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,0]])

   B = np.array([[dt * math.cos(x[2,0]), 0],
                 [dt * math.sin(x[2,0]), 0],
                 [0, dt],
                 [1,0]])
    
   x = A @ x + B @ u

   return x

def jacob_f(x, u):

    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -dt * v * math.sin(yaw), dt * math.cos(yaw)],
        [0.0, 1.0, dt * v * math.cos(yaw), dt * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def observation_model(x):

    z = H @ x 

    return z

def ekf_estimation(xEst, PEst, z, u):
    
    #Predict 
    xPred = state_model(xEst, u)
    #state covariance
    jF = jacob_f(xEst, u)
    PPred = jF*PEst*jF.T + Q

    #Update
    zPred = observation_model(xPred)

    y = z - zPred #measurement residual
    
    S = H @ PPred @ H.T + R #Innovation covariance

    K = PPred @ H.T @ np.linalg.inv(S) #kalman gain

    xEst = xPred + K @ y #updating state

    PEst = ((np.eye(len(xEst))) - K@H) @ PPred

    return xEst, PEst

def main():

    time = 0.0

    #state vector 
    xEst = np.zeros((4,1))
    xTrue = np.zeros((4,1))
    PEst = np.eye(4)
    
    #history
    hxEst = xEst
    hxTrue = xTrue 

    while True:

        frames = pipeline.wait_for_frames()

        raw_accel = frames[0].as_motion_frame().get_motion_data()
        raw_gyro = frames[1].as_motion_frame().get_motion_data()

        rs_to_base_tfm = np.asarray([[0,0,1],[1,0,0],[0,1,0]])#transfromation matrix

        accel = np.asarray([raw_accel.x, raw_accel.y, raw_accel.z])
        accel = np.transpose(np.matmul(rs_to_base_tfm, np.transpose(accel)))

        gyro = np.asarray([raw_gyro.x, raw_gyro.y, raw_gyro.z])
        gyro = np.transpose(np.matmul(rs_to_base_tfm, np.transpose(gyro)))

        #calculating net acceleration
        accel_net = math.sqrt((pow(accel[0],2) + pow(accel[1],2)))

        u = np.array([[accel_net*dt], [gyro[2]]]) #control input
        
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
            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            
            #plotting estimated state (represented by red line)
            plt.plot(hxEst[0, :], hxEst[1, :], "-r")
            
            plot.plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


     
            
if __name__ == '__main__':
    main()
