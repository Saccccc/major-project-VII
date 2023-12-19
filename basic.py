import numpy as np
import matplotlib.pyplot as plt
import fungsi as fn

M = 10 #number of ue
N = 3 # number of UAV and cluster
K = 10 #number of element of IRS
B = 1e6 #Bandwidth 1 Mhz
b0 = -30
P_temp = np.array((1,2,3))
P_lain = 4 #Pk+Pc watt
noise = 1
k_1 = 2
k_2 = 2.2
PkPc = 4

#Generate location of UEs in each cluster default
x_ues_temp = np.array((0,200,400,600),dtype=np.int_)
y_ues_temp = np.array((0,300,0,700),dtype=np.int_)
z_ues_temp = np.array((0,0,0,0),dtype=np.int_)
k1 = np.array((1,0),dtype=np.int_)
k2 = np.array((0,1),dtype=np.int_)
cov = 500 #radius

point = np.zeros((M,2,N),dtype=np.float32)
edge=np.linspace(0,1,200)*2*np.pi;
fig1, ax1 = plt.subplots()
for i in range(N):
    for i0 in range(M):
        r = np.random.rand(1,1)*cov
        theta = np.random.rand(1,1)*2*np.pi
        px = r*np.cos(theta)
        py = r*np.sin(theta)
        point_ = (x_ues_temp[i],y_ues_temp[i]) + (px*k1) + (py*k2)
        point[i0,:,i] = point_        

    ax1.scatter(point[:,0,i], point[:,1,i],marker='o')
    ax1.plot(x_ues_temp[i]+(cov*np.cos(edge)),y_ues_temp[i]+(cov*np.sin(edge)))   
plt.suptitle('Number of Cluster = '+str(N),fontsize='x-large', fontweight='bold')
plt.title('Number of UEs each Cluster = '+str(M), fontsize='large', fontweight='book')
plt.xlabel('x axist (m)')
plt.ylabel('y axist (m)')       
plt.grid(b=True, which='major')

#Generate location of UAVs & IRS default
x_uav_temp = np.array((0,200,400),dtype=np.float32)
y_uav_temp = np.array((0,300,0),dtype=np.float32)
z_uav_temp = np.array((200,200,200),dtype=np.float32)
irs_loc = np.array((500,500,30),dtype=np.float32)


#Generate the distance
d_nirs = np.zeros((N),dtype=np.float32)
d_irsm = np.zeros((N,M),dtype=np.float32)
for i in range(N):
    d_nirs[i] = np.sqrt((x_uav_temp[i]-irs_loc[0])**2 + (y_uav_temp[i]-irs_loc[1])**2 + (z_uav_temp[i]-irs_loc[2])**2)
    for i0 in range(M):
        d_irsm[i,i0] = np.sqrt((irs_loc[0]-point[i0,0,i])**2 + (irs_loc[1]-point[i0,1,i])**2 + (irs_loc[2])**2)



P = np.zeros(N,dtype = np.float32)
phi_ = np.zeros(K,dtype = np.complex128)
for i in range(len(action)):
    if i < N:
        P[i] = action[i]*env.power_normalize
    else:
        phi_[i-N] = np.exp(1j*(action[i]*env.phase_normalize))
phi = np.diag(phi_)


P = np.array((1,2,3))
theta = np.random.rand(K)*2*np.pi
phi_ = np.exp(1j*theta)
phi = np.diag(phi_)
phi_array = np.zeros(K,dtype=np.complex128)
h_nirs = np.zeros((1,K,N),dtype=np.complex128)
h_irsm = np.zeros((K,1,M,N),dtype=np.complex128)
it = 100
h_nlos = np.zeros((K,it),dtype=np.complex128)
for i in range(K):
    for i0 in range(it):
        h_nlos[i,i0] = np.random.randn(1)+(1j*(np.random.rand(1)))
    
h_nlos = np.mean(h_nlos,axis=1)
for i in range(K):
    phi_array[i] = np.exp((-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))/300)
    
for i in range(N):
    h_nirs[:,:,i] = np.sqrt(fn.db2lin(b0)*(np.power(env.d_nirs[i],-env.k1)))*phi_array*np.sqrt(0.5)
    
    for i0 in range(M):
        h_irsm[:,0,i0,i] = np.sqrt(fn.db2lin(b0)*np.power(env.d_irsm[i,i0],-env.k2)
                                  )*((np.sqrt(env.b1/(1+env.b1))*phi_array)+(np.sqrt(1/(2))*h_nlos))
        
sinr = np.zeros((N,M),dtype=np.float32)
throughput = np.zeros((N,M),dtype=np.float32)
Ptot = 0
s1 = np.zeros((N,M), dtype = np.complex128)
for i in range(N):
    for i0 in range(M):
        s = np.matmul(h_nirs[:,:,i],phi)
        s = np.matmul(s,h_irsm[:,:,i0,i])
        sinr_a = P[i]*np.power(np.abs(s),2)
        s1[i,i0] = s
        
        sinr_b = 0
        for k0 in range(N):
            if k0 != i:
                s = np.matmul(h_nirs[:,:,k0],phi)
                s = np.matmul(s,h_irsm[:,:,i0,k0])
                sinr_b += P[k0]*np.power(np.abs(s),2)
                
        sinr[i,i0] = sinr_a/(sinr_b+(env.noise))
        throughput[i,i0] = np.log2(1+sinr[i,i0])
        
    Ptot += P[i]
reward = np.sum(throughput)/(Ptot+PkPc)    
s1_reshape = s1.reshape(-1,1)
new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)




theta = np.random.rand(K)*2*np.pi
phi_ = np.exp(1j*theta)
phi = np.diag(phi_)
phi_array = np.zeros(K,dtype=np.complex128)
h_irsm = np.zeros((K,1,M),dtype=np.complex128)
 
for i in range( K):
    phi_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi)/100)
     
h_nirs = (np.sqrt(fn.db2lin(b0)*(np.power(env.d_nirs[0],-k1)))
           *phi_array).reshape((1,K))
 
for i0 in range(M): 
    h_irsm[:,0,i0] = np.sqrt(fn.db2lin(b0)*np.power(env.d_irsm[0,i0],-k2)
                                )*((np.sqrt(b1/(1+b1))*phi_array)+(np.sqrt(1/(2))*env.h_nlos))
 
s1 = np.zeros((M), dtype = np.complex128)
for i0 in range(M):
        s = np.matmul(h_nirs,phi)
        s = np.matmul(s,h_irsm[:,:,i0])
        s1[i0] = s
state = np.concatenate((np.real(s1),np.imag(s1)),axis=0)


