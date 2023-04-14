import rospy
import numpy as np
import math
from lib import observer, reference, ps4, u_data, ctrl_gains, tau, s_p
from math_tools import *
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

### Write your code here ###

def heading_control(psi, r, psi_d, psi_d_dot, psi_d_ddot, k1_psi):
    # Compute heading error
    z1_psi = psi - psi_d
    #Virtual control
    alpha_r = -k1_psi*z1_psi + psi_d_dot

    # Compute heading rate error
    z2_r = r - alpha_r

    z1_dot_psi = -k1_psi*z1_psi + z2_r

    alpha_dot_r = -k1_psi*z1_dot_psi + psi_d_ddot

    return alpha_r, alpha_dot_r, z2_r, z1_psi


def positional_control(psi, r, p, v, p_d, p_d_dot, p_d_ddot, s_dot, w, v_s, v_st, v_ss, K1):
    # Compute rotational matrix
    R = R2(psi)
    R_T = np.transpose(R)
    S = np.array([[0, -1],[1, 0]])
    R_dot = R@(r*S)
    R_dot_T = np.transpose(R_dot)
    # Compute positonal error
    z1_p = R_T@(p - p_d)
    alpha_v = -K1@z1_p +  R_T@p_d_dot*v_s
    z2_v = v - alpha_v
    z1_p_dot = -K1@z1_p - r*S@z1_p + z2_v - R_T@p_d_dot*w
    alpha_dot_v = -K1@z1_p_dot + R_dot_T@p_d_dot*v_s + R_T@p_d_ddot*v_s*s_dot + R_T@p_d_dot*(v_ss*s_dot + v_st)
    return alpha_v, alpha_dot_v, z2_v, z1_p
    #Virtual control


def clf_control_law(alpha, alpha_dot, z2, K2, b_hat, z1):
    M = np.array([[9.51, 0.0, 0.0], [0.0, 9.51, 0], [0.0, 0.0, 0.116]])  # Inertia matrix
    D = np.diag(np.array([1.96, 1.96, 0.196]))
    # xi_dot = z2 #integrator
    tau = -K2@z2 - b_hat + D@alpha + M@alpha_dot
    return tau

def fld_control_law(v, alpha_dot, z2, K2, b_hat):
    M = np.diag(np.array([9.51, 9.51, 0.116]))  # Inertia matrix
    D = np.diag(np.array([1.96, 1.96, 0.196]))
    #xi_dot = z2 #integrator
    tau = D@v - M@(K2@z2 + b_hat - alpha_dot)
    return tau

def consecrate(alpha_r, alpha_dot_r, alpha_v, alpha_dot_v, z2_v, z2_r, z1_p, z1_psi):
    z2 = np.array([[0],[0],[z2_r]])
    z1 = np.array([[0], [0], [z1_psi]])
    alpha = np.array([[0],[0],[alpha_r]])
    alpha_dot = np.array([[0],[0],[alpha_dot_r]])

    for i in range(0,1):
        z1[i][0] = z1_p[i]
        z2[i][0] = z2_v[i]
        alpha[i][0] = alpha_v[i]
        alpha_dot[i][0] = alpha_dot_v[i]
    return z1, z2, alpha, alpha_dot

def saturation(tau):
    F_max = 1                                              # [N]
    T_max = 0.3
    #Initialize output-array

    if (tau[0][0] == 0 and tau[1][0] == 0):                                        # [Nm]
        ck = F_max/(math.sqrt(tau[0][0]**2 + tau[1][0]**2 + 0.0001))
    else:
        ck = F_max/math.sqrt(tau[0][0]**2 + tau[1][0]**2)

    # Saturate surge and sway
    if (ck < 1):
         tau[0][0] = ck*tau[0][0]
         tau[1][0] = ck*tau[1][0]

    # Saturate yaw
    if  (np.abs(tau[2]) >= T_max):
        tau[2][0] = np.sign(tau[2])*T_max

    return tau

### End of custom code

def loop():
    # Extract controler gains, observer estimates and heading
    K1, K2, Ki, mu, U_ref = ctrl_gains.get_data()
    k1 = K1[2]
    K1 = np.diag(np.array([K1[0], K1[1]]))
    K2 = np.diag(K2)
    eta_hat, nu_hat, b_hat = observer.get_observer_data()
    p = np.take(eta_hat, [0, 1])
    p = p[:, np.newaxis]
    upsilon = np.take(nu_hat, [0, 1])
    upsilon = upsilon[:, np.newaxis]
    psi = rad2pipi(eta_hat[2])
    r = nu_hat[2]
    p_d, p_d_prime, p_d_prime2 = reference.get_ref()
    b_hat = b_hat[:, np.newaxis]

    # Get what we need from the desired references:
    psi_d = p_d[2]
    psi_d_dot = p_d_prime[2]
    psi_d_ddot = p_d_prime2[2]

    # Then we transform our position to the right form
    p_d = np.take(p_d, [0, 1])
    p_d = p_d[:, np.newaxis]
    p_d_prime = np.take(p_d_prime, [0, 1])
    p_d_prime = p_d_prime[:, np.newaxis]
    p_d_prime2 = np.take(p_d_prime2, [0, 1])
    p_d_prime2 = p_d_prime2[:, np.newaxis]


    # We retrieve our speed assignment and parameterization variable
    v_ts = 0
    s, s_dot = s_p.get_s()
    v_s, v_ss, w = reference.get_speed_assignemt()

    # Compute our errors and virtual control

    alpha_v, alpha_dot_v, z2_v, z1_p = positional_control(psi, r, p, upsilon, p_d, p_d_prime, p_d_prime2, s_dot, w, v_s, v_ts, v_ss, K1)
    alpha_r, alpha_dot_r, z2_r, z1_psi = heading_control(psi, r, psi_d, psi_d_dot, psi_d_ddot, k1)

    z1, z2, alpha, alpha_dot = consecrate(alpha_r, alpha_dot_r, alpha_v, alpha_dot_v, z2_v, z2_r, z1_p, z1_psi)

    tau_ctrl = clf_control_law(alpha, alpha_dot, z2, K2, b_hat, z1)

    tau_sat = saturation(tau_ctrl)
    tau_sat = np.array([tau_sat[0][0], tau_sat[1][0], tau_sat[2][0]])
    tau.publish(tau_sat)
    """
    All calls to functions and methods should be handled inside here. loop() is called by the main-function in ctrl_node.py
    """
