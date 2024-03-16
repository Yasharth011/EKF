#from time import sleep, time
    from math import sin, cos, tan, pi

# Filter coefficient
alpha = 0.1

phi_dot = 0.0
theta_dot = 0.0

def filter()        

    [bx, by, bz] = 

    # Get estimated angles from raw accelerometer data
    [phi_hat_acc, theta_hat_acc] =
    
    # Get raw gyro data and subtract biases
    [p, q, r] =
    p -= bx
    q -= by
    r -= bz
    
    # Calculate Euler angle derivatives 
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r
    
    # Update complimentary filter
    phi_hat = (1 - alpha) * (phi_hat + dt * phi_dot) + alpha * phi_hat_acc
    theta_hat = (1 - alpha) * (theta_hat + dt * theta_dot) + alpha * theta_hat_acc   
    

    return phi_hat, theta_hat
