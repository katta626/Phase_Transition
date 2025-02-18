import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import minimize

@jit(nopython=True)
def P_from_nb(n_list, K, Gam):
    p = K*(n_list)**(Gam)
    return p

@jit(nopython=True)
def mu_from_nb(n_list, K, Gam, m0):
    t1 = ((K*Gam)/(Gam - 1))*((n_list)**(Gam - 1))
    re = t1 + m0
    return re

@jit(nopython=True)
def e_from_n(n_list, K, Gam, m0):
    t1 = (K * ((n_list) ** Gam)) / (Gam - 1)
    t2 = t1 + (m0 * (n_list))
    return t2

@jit(nopython=True)
def cs_from_n(mu,K,Gam,n_list):
    t1 = 1/mu
    t2 = K*Gam*((n_list)**(Gam-1))
    t = t1*t2
    return t

@jit(nopython=True)
def k(p_initial,n_initial,gam):
    result = p_initial/(n_initial**gam)
    return result

@jit(nopython=True)
def calculate_m0(energy_density, K, n, Gam):
    return (energy_density - (K * (n ** Gam)) / (Gam - 1)) / n

n0 = 0.16

def ntt(z):
    return np.linspace(1.2*0.16,10*0.16,z)

def delta_nt(nt,z):
    return np.linspace(0*nt,9.09*nt,z)

def gamm_h(z):
    return np.linspace(1.1,4,z)
def gamm_q(z):
    return np.linspace(1.1,4,z)

def seidov_limit_constraint(delta_epsilon, epsilon_c, P_c):
    lhs = delta_epsilon / epsilon_c
    rhs = 0.5 + 1.5 * (P_c / epsilon_c)
    
    # Return True if lhs is greater than or equal to rhs, else False
    return lhs >= rhs

import itertools

def constructing_EOS(n, y):

    p, mu, e, cs = [], [], [], []
    p_c_total, mu_c_total = [], []
    gam_h, gam_q =[], []
    n_transition, delta_n = [], []
    z = n // 2
    
    gamma_h = gamm_h(z)
    gamma_q = gamm_q(z)

    #gamma_h = [4.921]  # Instead of gamm_h(z)
    #gamma_q = [4.0] 
    #gamma_h = [2.50,2.75,3.0,3.25,3.50]
    nt = ntt(z)
    #print(nt)
    # Generate all combinations of gamma_h, gamma_q, nt, delta_n using itertools.product
    for g_h, g_q, n_t in itertools.product(gamma_h, gamma_q, nt):
        delta_n_11 = []
        #print("n")

        # Generate delta_n for each combination of n_t
        d_n_list = delta_nt(n_t, z)
        #d_n_list = [0.1]
        
        # Iterate over delta_n values
        for d_n in d_n_list:
            #print(d_n)
            pih = 3.542
            muih = 977.5
            eih = 1.1 * 0.16 * muih - pih
            k_had = pih / (1.1 * 0.16) ** g_h

            n_had = np.linspace(1.1 * 0.16, n_t, y)

            p_had = P_from_nb(n_had, k_had, Gam=g_h)
            m_had = calculate_m0(energy_density=eih, K=k_had, n=1.1*0.16, Gam=g_h)
            mu_had = mu_from_nb(n_list=n_had, K=k_had, Gam=g_h, m0=m_had)
            e_had = e_from_n(n_list=n_had, K=k_had, Gam=g_h, m0=m_had)
            cs_had = cs_from_n(mu=mu_had, K=k_had, Gam=g_h, n_list=n_had)

            if cs_had[-1] < 1:

                n_mid = np.linspace(n_t, n_t + d_n, y)

                p_mid = np.linspace(p_had[-1], p_had[-1], y)
                mu_mid = np.linspace(mu_had[-1], mu_had[-1], y)
                e_mid = n_mid * mu_mid - p_mid
                cs_mid = np.gradient(p_mid, e_mid)

                delta_e = e_mid[-1] - e_mid[0]
                p_c = p_had[-1]
                mu_c = mu_had[-1]
                e_c = e_had[-1]
                
                truth_value = True  # Assuming Seidov limit is always satisfied for now

                if truth_value:
                    n_iq = n_t + d_n
                    piq = p_had[-1]
                    k_quark = piq / (n_iq ** g_q)
                    m_quark = calculate_m0(energy_density=e_mid[-1], K=k_quark, n=n_iq, Gam=g_q)
                    n_fq = (1000 / k_quark) ** (1 / g_q)

                    n_quark = np.linspace(n_t + d_n, n_fq, y)

                    p_quark = P_from_nb(n_quark, k_quark, Gam=g_q)
                    mu_quark = mu_from_nb(n_list=n_quark, K=k_quark, Gam=g_q, m0=m_quark)
                    e_quark = e_from_n(n_list=n_quark, K=k_quark, Gam=g_q, m0=m_quark)
                    cs_quark = cs_from_n(mu=mu_quark, K=k_quark, Gam=g_q, n_list=n_quark)

                    # Add this condition:
                    if cs_quark[-1] < 1:
                        p.append(np.concatenate((p_had, p_mid, p_quark)))
                        mu.append(np.concatenate((mu_had, mu_mid, mu_quark)))
                        e.append(np.concatenate((e_had, e_mid, e_quark)))
                        cs.append(np.concatenate((cs_had, cs_mid, cs_quark)))

                        gam_h.append(g_h)
                        gam_q.append(g_q)

                        n_transition.append(nt)
                        delta_n_11.append(d_n)

                        p_c_total.append(p_c)
                        mu_c_total.append(mu_c)
                delta_n.append(delta_n_11)
    # Default return if no valid result found
    return p, mu, e, cs, mu_c_total, p_c_total, gam_h, gam_q, n_transition, delta_n