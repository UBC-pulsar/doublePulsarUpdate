#!/usr/bin/env python
# Author: Chiara M. F. Mingarelli, mingarelli@gmail.com.

import math
import cmath
import matplotlib.pyplot as plt
from matplotlib import rc, text
from matplotlib.font_manager import FontProperties
import numpy as np
import scipy.special as sp
from scipy.integrate import quad, dblquad
import sys #to enable command line m,l
from sys import stdout 


fontP = FontProperties()
rc('text', usetex=True)
plt.rcParams['font.size'] = 15

sqrt=math.sqrt
cos=math.cos
sin=math.sin
exp=cmath.exp # make sure to use cmath for complex exponents
pi=math.pi
log=math.log
atan=math.atan
j=complex(0,1)


#General formulas for checking

def omega_dot(mp,mc,ecc,Pb):
    """
    Advance of the periastron. Use Pb in seconds for consistent units.
    mp and mc have units of solar masses
    reproduces KW09 (This is Eq 10)
    """
    ans=3*T_sun_KW09**(2./3)*(Pb/(2*pi))**(-5./3)*(mp+mc)**(2./3)/(1-ecc*ecc)
    return ans

def pb_dot(mp,mc,ecc,Pb):
    """
    Time derivative of the orbital period.
    mp and mc have units of solar masses
    reproduces KW09 (This is Eq 14)
    """
    pbdot=(-192.*pi/5)*(T_sun_KW09**(5./3))*((Pb/(2*pi))**(-5./3))*fe(ecc)*mp*mc/((mp+mc)**(1./3))
    return pbdot

def spinOrbit_p(mp,mc,ecc,Pb):
    """
    Spin-orbit coupling of m_p.
    mp and mc have units of solar masses
    reproduces KW09 Table2 GR expected value for B.
    This is Eq 15.
    """
    ans=T_sun_KW09**(2./3)*(2*pi/Pb)**(5./3)*mc*(4*mp+3*mc)/(2*(mp+mc)**(4./3)*(1.-ecc*ecc))
    return ans

#Functions to update

def M_tot(m1,m2):
    """
    Total mass
    """
    ans=m1+m2
    return ans

def x1(m1,m2):
    """
    Damour and Schafer 1988, p.158
    """
    ans=m1/M_tot(m1,m2)
    return ans

def x2(m1,m2):
    """
    Damour and Schafer 1988, p.158
    """
    ans=1.-x1(m1,m2)
    return ans

def betaO(m1,m2,nb):
    """
    Damour and Schafer 1988, Eq 5.19 p. 159
    """
    ans=(G*M_tot(m1,m2)*nb)**(1./3)/c
    return ans

def betaS(S_a,m_a):
    """
    Damour and Schafer 1988, Eq 5.20 p. 159, written in terms
    of the spin parameter S_a.
    """
    #ans=c*S_a/(G*m_a*m_a)
    ans=0.01
    return ans

def fO(e_T,m1,m2):
    """
    Damour and Schafer 1988, Eq 5.21 p. 159
    """
    ans=(1./(1-e_T*e_T))*(39./4*x1(m1,m2)*x1(m1,m2)+27./4*x2(m1,m2)*x2(m1,m2)+15*x1(m1,m2)*x2(m1,m2))
    - (13./4*x1(m1,m2)*x1(m1,m2)+.25*x2(m1,m2)*x2(m1,m2)+13./3*x1(m1,m2)*x2(m1,m2))
    return ans

def g_S(m1,m2,ecc,Sini,Cosi,K_0vec,kvec,svec):
    """
    Damour and Schafer 1988, Eq 5.22 p. 159
    """
    num=x1(m1,m2)*(4*x1(m1,m2)+3*x2(m1,m2))*((3*Sini*Sini-1.)*np.dot(kvec,svec)+Cosi*np.dot(K_0vec,svec))
    deno=6.*sqrt(1-ecc*ecc)*sini*sini
    ans=num/deno
    return ans

def g_S_parallel(m1,m2,ecc):
    """
    From above, assuming k*s=1, i.e. is parallel.
    This means that i=90 degrees and sin(i)=1, cos(i)=0.
    This can be used to estimate a limit on g_S.
    """
    g_par=(4*x1(m1,m2)*x1(m1,m2)/3.+x1(m1,m2)*x2(m1,m2))/sqrt(1.-ecc*ecc)
    return g_par

def k_tot(m1,m2,ecc,Sini,Cosi,nb,K_0vec,kvec,svec1,svec2,SpinA,SpinB):
    """
    Damour and Schafer 1988, Eq 5.23 p. 159
    """
    prefactor=3*betaO(m1,m2,nb)*betaO(m1,m2,nb)/(1.-ecc*ecc)
    func=1.+fO(e_T,m1,m2)*betaO(m1,m2,nb)*betaO(m1,m2,nb)
    -g_S(m1,m2,ecc,sini,cosi,K_0vec,kvec,svec1)*betaO(m1,m2,nb)*betaS(Spin1,m1)
    -g_S(m1,m2,ecc,sini,cosi,K_0vec,kvec,svec2)*betaO(m1,m2,nb)*betaS(Spin2,m1)
    return prefactor*func

def omega_tot(eT):
    """
    total periastron advance w_tot= w_1PN+w_1.5PN+w_2PN
    """
    ans=3*betaO()*betaO()*nb/(1-eT*eT)*(1+fO()*betaO()*betaO()-gS()*betaO()*betaS())
    return ans

def fe(ecc):
    """
    Kramer and Wex 2009, Eq 16 (p. 9)
    """
    fe=(1.+(73./24)*ecc*ecc+(37./96)*ecc**4)/(1-ecc*ecc)**(7./2)
    return fe

if __name__ == "__main__":
    
    #Physical Constants
    year = 86400*365.24218967 # seconds in a year 
    c=2.99792458e8
    G=6.67428e-11
    s_mass_secs=G*(1.98892e30)/(c**3) #in seconds
    M_sun=1.98892e30      # kilograms
    T_sun= G*M_sun/c**3   # note, this yeilds 4.9267398258e-06, which differs from KW09
                          # who claim T_sun=4.925490947e-06. This results in a 
                          # fractional difference of 0.000253489901926 or ~0.025%

    #Constants from Kramer and Wex 09
    mc = 1.3381                 # mass of A [solar masses]
    mp = 1.2489                 # mass of B [solar masses]
    Pb_day = 0.102251562485     # orbital period in units of days 0.102 251 562 48(5)
    Pb_sec = Pb_day*86400.      # converting days to seconds
    ecc = 0.08777759            # 0.087 777 5(9)
    M = 2.5870816               # total system mass 2.58708(16) [solar masses].
    nb = 2*pi/Pb_day            # orbital frequency, units of days
    omegaDot_KW09 = 16.8994768  # advance of the periastron degrees/year 16.899 47(68)
    T_sun_KW09 = 4.925490947e-6
    Pb_dot = -1.25217e-12       # -1.252(17)e-12, from Kramer et al (2006), Science (typo in KW09).
    Pb_dot_GR = -1.24787e-12    # GR prediction from KW09 is 1.24787(13)
    SO_coupling = 5.07347       # GR prediction is 5.0734(7)
    Sini=0.99987                # GR prediction
    Cosi=sqrt(1.-Sini*Sini)     # Trig formula applied

    #Testing Functions
    #K_0vec=np.array(1,0,0)
    #kvec=np.array(1,0,0)
    #svec1=np.array(1,0,0)
    #svec2=np.array(1,0,0)
    SpinA=0.5
    SpinB=0.5

    #print "Omega dot diff is", "%.4e" %(omega_dot(mp,mc,ecc,Pb_sec)*year*180/pi - omegaDot_KW09), "degrees/year."
    #yields correct value up to 10^-4... still not great
                
    #print "Pb dot (GR) diff is", "%.4e" %(pb_dot(mp,mc,ecc,Pb_sec) - Pb_dot_GR)
    #yields correct value, diff is 10^-17 for GR prediction.

    #print "SO coupling diff for B is", "%.4e" %  (spinOrbit_p(mp,mc,ecc,Pb_sec)*year*180/pi - SO_coupling), "degrees/year."

    #print "testing k_total", k_tot(m1,m2,ecc,Sini,Cosi,nb,K_0vec,kvec,svec1,svec2,SpinA,SpinB)

    print "testimating g_sA", g_S_parallel(mp,mc,ecc)
