# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:04:09 2021

@author: PC
"""
from functools import partial
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Evolutionary model

#Parameters 
b = 1
p = 0.27
w = 0.5

a0= 0.01
W0 = 1
q0 = 0.005

zb = 10
hs = 0
hf = 0.2
r = 5
l = 0.1


z = zb
time = np.linspace(0, 1000, 100)

#Defining function:
def a(z,zm):
    a = 2*a0 * np.exp(b*(z-zm))/(1+np.exp(b*(z-zm)) )
    return(a)

def Nu(z):
    Nu = (Nu0 / (u* np.sqrt(2*np.pi))) * np.exp(-(zb-z)**2/u**2)
    return Nu

def n(z, zm, S,F,Sm):
    res = Nu(z)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def nf(z, zm, S,F,Sm):
    res = Nu(0)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def nm(zm, z, S,F,Sm):
    res = Nu(zm)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def q(z):
    q = q0* np.exp(p*z)
    return q

def W(z):
    W = W0*np.exp(w*z)- W0
    return W

def f(pop,t,z,zm):
    S = pop[0]
    F = pop[1]
    Sm = pop[2]
    f0 =  r*S*n(z,zm,S,F,Sm)/(n(z,zm,S,F,Sm) +hs)* 1/(1+ a(z,z)*S+a(z,zm)*Sm+ a(z,0)*F+W(z)) - l *S
    f1 =  r*F*nf(z,zm,S,F,Sm)/(nf(z,zm,S,F,Sm) + hf)* 1/(1+ a(0,z)*S+a(0,zm)*Sm+ a(0,0)*F+W(0)) - l *F
    f2= r*Sm*nm(zm,z,S,F,Sm)/(nm(zm,z,S,F,Sm) +hs)* 1/(1+ a(zm,z)*S+a(zm,zm)*Sm+ a(zm,0)*F+W(zm)) - l*Sm
    f = [f0, f1,f2]
    return f

############################### Versions depending on Nu0 and u

def nm2(zm, z, S,F,Sm, Nu0, u):
    res = Nu2(zm,Nu0, u)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def n2(z, zm, S,F,Sm, Nu0, u):
    res = Nu2(z, Nu0, u)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def nf2(z, zm, S,F,Sm,Nu0, u):
    res = Nu2(0, Nu0, u)/(1 + q(z)*S + q(zm)*Sm + q(0)*F)
    return res

def Nu2(z,Nu0, u):
    nu = (Nu0 / (u* np.sqrt(2*np.pi))) * np.exp(-(zb-z)**2/u**2)
    return nu


def f2(pop,t,z,zm,Nu0,u):
    S = pop[0]
    F = pop[1]
    Sm = pop[2]
    f0 =  r*S*n2(z,zm,S,F,Sm, Nu0, u)/(n2(z,zm,S,F,Sm, Nu0, u) +hs)* 1/(1+ a(z,z)*S+a(z,zm)*Sm+ a(z,0)*F+W(z)) - l *S
    f1 =  r*F*nf2(z,zm,S,F,Sm, Nu0, u)/(nf2(z,zm,S,F,Sm, Nu0, u) + hf)* 1/(1+ a(0,z)*S+a(0,zm)*Sm+ a(0,0)*F+W(0)) - l *F
    f2= r*Sm*nm2(zm,z,S,F,Sm, Nu0, u)/(nm2(zm,z,S,F,Sm, Nu0, u) +hs)* 1/(1+ a(zm,z)*S+a(zm,zm)*Sm+ a(zm,0)*F+W(zm)) - l*Sm
    f = [f0, f1,f2]
    return f

###################Equilibrium of F ####################################

def Feq(Nu0, u):
    NU = Nu2(0,Nu0, u)
    A =l*q0*hf*a0/NU
    B = l*(a0 + hf*a0/NU+ hf*q0/NU)
    C = l + l*hf/NU - r

    Fe = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
    return Fe


Nu_val = np.linspace(14,500,100)
u_val = np.linspace(4,10,100)

X, Y = np.meshgrid(Nu_val, u_val)

###################### Defining intervals of u and Nu0 for which F_eq is > 0  ######################
Z = Feq(X, Y)

plt.pcolor(X, Y, Z, cmap='tab20', shading = 'auto')
plt.xlabel('Nu0')
plt.ylabel('u')
plt.colorbar()
plt.show()

Nu0 = 200
u =7 

F  = Feq(Nu0, u)
################################## Growth rate of SM ##############################################
def taux_sm(zm):
    taux = r*nm(zm,z,0,F,0)/(nm(zm,z,0,F,0) +hs)* 1/(1+ a(zm,0)*F+W(zm)) - l
    return taux


plt.plot(np.linspace(0,10,100), taux_sm(np.linspace(0,10,100)))
plt.xlabel("trait zm")
plt.title("taux de croissance de Sm (u=7 Nu0 =200)")
plt.grid(True)
plt.show()

################################# Evolution of the system depending on the trait zm ####################
popInit = [0,F,1]

for i in np.linspace(0,10,100):
    zm = i 
    traj = odeint(partial(f, z=z,zm=zm), y0= popInit, t=time)
    plt.scatter(i, traj[-1,2], label = 'Sm')
    plt.xlabel('zm')
    plt.title('Densité de population Sm u=7 Nu0 = 20 ')

for i in np.linspace(0,10,100):
    zm = i 
    traj = odeint(partial(f, z=z,zm=zm), y0= popInit, t=time)
    plt.scatter(i, traj[-1,1], label = 'F')
    plt.xlabel('zm')
    plt.title('Densité de population F u=7 Nu0 = 20 ')
    
##### Ploting a 2D diagram to see the areas where (for zm =5) there's coexistance between F and Sm AND where Sm outcompetes F AND where Sm never grows  
zm = 5

FF = np.eye(100)
SM = np.eye(100)
for i in range(100):
    for j in range(100): 
        y = u_val[99 - i]
        x = Nu_val[j]  
        traj = odeint(partial(f2, z=z,zm=zm, Nu0 =x, u =y), y0= [0, Feq(x,y), 1], t=time)
        FF[99 - i,j]= traj[-1,1]
        SM[99 - i,j] = traj[-1,2]
        
######### Coexistance matrix
coex = np.eye(100)
for i in range(100):
    for j in range(100): 
        if FF[i,j] > 10:
            if SM[i,j] > 10:
                coex[i,j] =2
            else:
                coex[i,j] = 0
        else:
            coex[i,j] = 1

plt.pcolor(X, Y, coex, cmap='plasma' , shading = 'auto')
plt.xlabel('Nu0')
plt.ylabel('u')
plt.title('Zone de coexistance de F et SM (jaune), SM remplace F (rose) et dominance de F (bleu) pour zm = 5')
plt.show()
########################### Eolution of the system for zm = 5  ###############################

traj = odeint(partial(f, z=z,zm=zm), y0= popInit, t=time)


plt.plot(time, traj[:,0], label= "S")
plt.plot(time, traj[:,1], label= "F")
plt.plot(time, traj[:,2], label= "Sm")
plt.legend(loc="upper right")
plt.xlabel('time')
plt.show()

    
#####################  Equilibruim of Sm and F  ####################################

A = q(zm)*hs*l*a0/Nu(zm)
B = l/Nu(zm)*(Nu(zm)*a0+hs*a0+ q(zm)*hs +q(zm)*hs*W(zm))
C = l/Nu(zm)*(Nu(zm)+ Nu(zm)*W(zm)+hs+hs*W(zm)) -r

Sm = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
F = traj[-1,1]

popInit = [1,F,Sm]
############################Selection gradient in zm = 5 (direction of the adaptive dynamics)#########################################
def taux_s2(z,zm, Nu0, u):
    taux = r*n2(z,zm,0,F,0, Nu0, u)/(n2(z,zm,0,F,0, Nu0, u) +hs)* 1/(1+ a(z,zm)*Sm + a(z,0)*F +W(z)) - l
    return taux

H = 1e-5

def der_tauxS(z,zm, Nu0, u):
    der = (taux_s2(z+H,zm, Nu0, u) - taux_s2(z,zm, Nu0, u))/H
    return der

################### Drawing the selection gradient depending on Nu0 and u #############################
plt.plot(np.linspace(0,500,100), der_tauxS(zm,zm, np.linspace(0,500,100),u))
plt.xlabel('Nu0')
plt.title('Gradient de sélection en zm=5 en fonction de Nu0 (u=7)')

plt.plot(np.linspace(0.5,10,100), der_tauxS(zm,zm,Nu0, np.linspace(0.5,10,100)))
plt.xlabel('u')
plt.title('Gradient de sélection en zm=5 en fonction de u (Nu0 = 500)')