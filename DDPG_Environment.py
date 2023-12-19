import import_ipynb
import tools
import numpy as np
import matplotlib.pyplot as plt 
from tools import fungsi as fn

# defining environment class
class SA_DDPG_Env():
    def __init__(self,N,M,K,B,noise,b0,b1,k1,k2,PkPc,power_properties,phase_properties,
                 x_uav,y_uav,z_uav,irs_loc,UEs_loc):
        self.N = N
        self.M = M
        self.K = K
        self.B = B
        self.noise = noise
        self.b0 = b0
        self.b1 = b1
        self.k1 = k1
        self.k2 = k2
        self.PkPc = PkPc
        self.x_uav = x_uav
        self.y_uav = y_uav
        self.z_uav = z_uav
        self.irs_loc = irs_loc
        
        self.point = UEs_loc
        self.d_nirs, self.d_irsm = self.get_d()
        self.power_normalize = np.max(power_properties) - np.min(power_properties)
        self.phase_normalize = np.max(phase_properties) - np.min(phase_properties) 
        self.power_properties = power_properties/self.power_normalize
        self.phase_properties = phase_properties/self.phase_normalize
        self.power_space = space_generator(low=self.power_properties[0],high=self.power_properties[1])
        self.phase_space = space_generator(low=self.phase_properties[0],high=self.phase_properties[1])     
        self.min_action, self.max_action = self.get_bound_action()
        self.action_space = self.N+self.K
        self.observation_space = 2*(self.N*self.M)
        self.phi_array = self.get_phi_array()
        self.h_nlos = np.random.randn(self.K)+(1j*(np.random.randn(self.K)))
    
    # generates an array of complex numbers
    def get_phi_array(self):
        phi_array = np.zeros(self.K,dtype=np.complex128)
        for i in range(self.K):
            phi_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))
            
        return phi_array
    
    def reset_state(self):
        theta = np.random.rand(self.K)*2*np.pi
        phi_ = np.exp(1j*theta)
        phi = np.diag(phi_)
        
        h_nirs = np.zeros((1,self.K,self.N),dtype=np.complex128)
        h_irsm = np.zeros((self.K,1,self.M,self.N),dtype=np.complex128)
        
            
        for i in range(self.N):
            h_nirs[:,:,i] = np.sqrt(fn.db2lin(self.b0)*(np.power(self.d_nirs[i],-self.k1)))*self.phi_array
            
            for i0 in range(self.M): 
                h_irsm[:,0,i0,i] = np.sqrt(fn.db2lin(self.b0)*np.power(self.d_irsm[i,i0],-self.k2)
                                          )*((np.sqrt(self.b1/(1+self.b1))*self.phi_array)+(np.sqrt(1/(2))*self.h_nlos))
        
        s1 = np.zeros((self.N,self.M), dtype = np.complex128)
        for i in range(self.N):
            for i0 in range(self.M):
                s = np.matmul(h_nirs[:,:,i],phi)
                s = np.matmul(s,h_irsm[:,:,i0,i])
                s1[i,i0] = s
        s1_reshape = s1.reshape(-1,1)
        state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)
        
        return state[:,0]
    
    
    def get_bound_action(self):
        cntr = 0
        max_action = np.zeros((self.N+self.K),dtype=np.float32)
        min_action = np.zeros((self.N+self.K),dtype=np.float32)
        for i in range((self.N+self.K)):
            if cntr > (self.N-1):
                min_action[cntr] = self.phase_space.low
                max_action[cntr] = self.phase_space.high
            else:
                min_action[cntr] = self.power_space.low
                max_action[cntr] = self.power_space.high
            cntr += 1
        
        return min_action, max_action
    
    
    def get_d(self):
        d_nirs = np.zeros((self.N),dtype=np.float32)
        d_irsm = np.zeros((self.N,self.M),dtype=np.float32)
        for i in range(self.N):
            d_nirs[i] = np.sqrt((self.x_uav[i]-self.irs_loc[0])**2 + (self.y_uav[i]-self.irs_loc[1])**2 + 
                                (self.z_uav[i]-self.irs_loc[2])**2)
            for i0 in range(self.M):
                d_irsm[i,i0] = np.sqrt((self.irs_loc[0]-self.point[i0,0,i])**2 + 
                                       (self.irs_loc[1]-self.point[i0,0,i])**2 + (self.irs_loc[2])**2)
            
        return d_nirs,d_irsm


def step(self, action):
    P = np.zeros(self.N, dtype=np.float32)
    phi_ = np.zeros(self.K, dtype=np.complex128)

    # Extract power and phase values from the action vector
    for i in range(len(action)):
        if i < self.N:
            P[i] = action[i] * self.power_normalize
        else:
            phi_[i - self.N] = np.exp(1j * (action[i] * self.phase_normalize))
    phi = np.diag(phi_)

    h_nirs = np.zeros((1, self.K, self.N), dtype=np.complex128)
    h_irsm = np.zeros((self.K, 1, self.M, self.N), dtype=np.complex128)

    # Channel modeling for RIS and UAV-aided transmission modes
    # ... (existing code)

    # RSMA Signaling for Hybrid RIS-UAV-aided Transmission
    s_common = np.zeros((self.N, self.M), dtype=np.complex128)
    s_private = np.zeros((self.N, self.M), dtype=np.complex128)
    s_combined = np.zeros((self.N, self.M), dtype=np.complex128)

    # Aerial Pathloss Model
    beta_ij = np.zeros((self.N, self.M), dtype=np.float32)

    for i in range(self.N):
        for i0 in range(self.M):
            # Calculate distance between devices i and j (in this case, UAV and UE)
            d_ij = np.sqrt((self.x_uav[i] - self.point[i0, 0, i]) ** 2 +
                           (self.y_uav[i] - self.point[i0, 1, i]) ** 2 +
                           (self.z_uav[i] - self.irs_loc[2]) ** 2)

            # Calculate position-dependent pathloss exponent
            lambda_ij = np.arctan(np.sqrt((self.z_uav[i] - self.point[i0, 1, i]) ** 2) /
                                  ((self.x_uav[i] - self.point[i0, 0, i]) ** 2 +
                                   (self.y_uav[i] - self.point[i0, 1, i]) ** 2))

            a_lambda_ij = self.calculate_pathloss_exponent(lambda_ij)  # Implement calculate_pathloss_exponent function

            # Calculate large-scale fading term (beta_ij)
            beta_ij[i, i0] = self.calculate_large_scale_fading(d_ij, a_lambda_ij)  # Implement calculate_large_scale_fading function

            # Continue with RSMA Signaling for Hybrid RIS-UAV-aided Transmission
            s_common[i, i0] += np.matmul(h_nirs[:, :, i], phi)
            s_common[i, i0] += np.matmul(s_common[i, i0], h_irsm[:, :, i0, i])

            s_private[i, i0] += np.matmul(h_nirs[:, :, i], phi)
            s_private[i, i0] += np.matmul(s_private[i, i0], h_irsm[:, :, i0, i])

            # Combine common and private messages for Hybrid RIS-FD-UAV
            s_combined[i, i0] = s_common[i, i0] + s_private[i, i0]

    # Transmit the combined signal from BS to GUs using RIS and UAV
    # ... (Hybrid RIS-FD-UAV-aided transmission)

    # Decode common message at each user using SC
    sinr_common = np.zeros((self.N, self.M), dtype=np.float32)
    sinr_private = np.zeros((self.N, self.M), dtype=np.float32)
    sinr_combined = np.zeros((self.N, self.M), dtype=np.float32)

    for i in range(self.N):
        for i0 in range(self.M):
            # SINR to decode common message at UAV
            psi_i = np.abs(s_combined[i, i0]) ** 2
            psi_i_common = np.abs(s_common[i, i0]) ** 2
            psi_i_private = np.abs(s_private[i, i0]) ** 2

            sinr_common[i, i0] = psi_i_common / (psi_i + self.noise)

            # SINR to decode private message at UAV
            psi_i_private /= (psi_i + self.noise)
            for k0 in range(self.N):
                if k0 != i:
                    psi_i_private += (P[k0] * np.abs(s_combined[k0, i0]) ** 2) / (psi_i + self.noise)

            sinr_private[i, i0] = psi_i_private

            # SINR for common and private messages combined
            sinr_combined[i, i0] = psi_i / (psi_i_common + psi_i_private + self.noise)

    # Calculate rewards based on SINR values
    reward_common = np.sum(np.log2(1 + sinr_common))
    reward_private = np.sum(np.log2(1 + sinr_private))
    reward_combined = np.sum(np.log2(1 + sinr_combined))
    total_reward = reward_common + reward_private + reward_combined

    # Update state
    s_combined_reshape = s_combined.reshape(-1, 1)
    new_state = np.concatenate((np.real(s_combined_reshape), np.imag(s_combined_reshape)), axis=0)

    # Calculate throughput
    throughput_common = np.sum(np.log2(1 + sinr_common))
    throughput_private = np.sum(np.log2(1 + sinr_private))
    throughput_combined = np.sum(np.log2(1 + sinr_combined))
    total_throughput = throughput_common + throughput_private + throughput_combined

    return total_reward, new_state[:, 0], total_throughput

# Outside the class, add the following functions:
def calculate_pathloss_exponent(lambda_ij):
    # Constants (you may replace these with your actual values)
    q_ij = 1.0
    u_ij = 0.5
    psi_ij = 0.2
    psi_hat_ij = 0.3

    # Calculation of position-dependent pathloss exponent
    return (1 / (1 + np.exp(psi_ij * np.exp(psi_hat_ij * (lambda_ij - psi_ij))))) * q_ij + u_ij


def calculate_large_scale_fading(d_ij, a_lambda_ij):
    # Constants (you may replace these with your actual values)
    beta_o = 1.0  # Your channel gain at reference distance
    a = 2.0  # Your calculation for a

    # Calculation of large-scale fading term (beta_ij)
    return beta_o * np.power(d_ij, -a)

            
       
class UEs():
    
    def __init__(self,N,M,x_ues,y_ues,coverage):
        self.N = N
        self.M = M
        self.x_ues = x_ues
        self.y_ues = y_ues
        self.cov = coverage
        self.point = self.get_location()
    
    def get_location(self):
        point = np.zeros((self.M,2,self.N),dtype=np.float32)
        k1 = np.array((1,0),dtype=np.int_)
        k2 = np.array((0,1),dtype=np.int_)
        for i in range(self.N):
            for i0 in range(self.M):
                r = np.random.rand(1,1)*self.cov
                theta = np.random.rand(1,1)*2*np.pi
                px = r*np.cos(theta)
                py = r*np.sin(theta)
                point_ = (self.x_ues[i],self.y_ues[i]) + (px*k1) + (py*k2)
                point[i0,:,i] = point_        

        return point
    
    def plot_location(self):
        edge=np.linspace(0,1,200)*2*np.pi;
        fig1, ax1 = plt.subplots()
        for i in range(self.N):
            ax1.scatter(self.point[:,0,i], self.point[:,1,i],marker='o')
            ax1.plot(self.x_ues[i]+(self.cov*np.cos(edge)),self.y_ues[i]+(self.cov*np.sin(edge)))   
        plt.suptitle('Number of Cluster = '+str(self.N),fontsize='x-large', fontweight='bold')
        plt.title('Number of UEs each Cluster = '+str(self.M), fontsize='large', fontweight='book')
        plt.xlabel('x axist (m)')
        plt.ylabel('y axist (m)')       
        plt.grid(b=True, which='major') 
        
        

class OUNoise(object):
    def __init__(self, env, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.01, decay_period=1000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = env.action_space
        self.low          = env.min_action
        self.high         = env.max_action
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
      
    
    
class space_generator:
    def __init__ (self, low, high, dtype=np.float32):
        if np.isscalar(low) or np.isscalar(high):
            self.low = np.array([low]).astype(dtype)
            self.high = np.array([high]).astype(dtype)
        else:
            self.low = np.array(low).astype(dtype)
            self.high = np.array(high).astype(dtype)
        assert self.low.shape == self.high.shape, "low.shape doesn't match high.shape"
    
