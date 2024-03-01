import sys
import math
import numpy as np
import plot
import matplotlib.pyplot as plt

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

SIM_TIME = 50.0 #simulation time

show_animation = True

def calc_input():
    v = 1.0
    a = 0.1
    u = np.array([[v],[a]]) #control input

    return u

def observation(xTrue, xd, u):
    xTrue = state_model(xTrue, u)

    #adding noise to input
    ud = u + input_noise @ np.random.randn(2,1)

    xd = state_model(xd, ud)

    return xTrue, xd, ud

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
    
    xDR = np.zeros((3,1)) #dead reckoning

    #history
    hxEst = xEst
    hxTrue = xTrue 
    hxDR = xTrue 
    hz = np.zeros((3,1)) 

    while SIM_TIME>=time:
        
        time+= dt

        u = calc_input()

        xTrue, XDR, ud = observation(xTrue, xDR, u)

        z = observation_model(xTrue)

        xEst, Pest = ekf_estimation(xEst, PEst, z, ud)

        #store data histroy 
        hxEst = np.hstack((hxEst, xEst))
        hxTrue = np.hstack((hxTrue, xTrue))
        hxDR = np.hstack((hxDR, xDR))
        hz = np.hstack((hz,z))
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")

            """
            plt.text(0.45, 0.85, f"True Velocity Scale Factor: {true_scale_factor:.2f}", ha='left', va='top', transform=plt.gca().transAxes)
           plt.text(0.45, 0.95, f"Estimated Velocity Scale Factor: {estimated_scale_factor:.2f}", ha='left', va='top', transform=plt.gca().transAxes)
            """

            plot.plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
            
if __name__ == '__main__':
    main()
