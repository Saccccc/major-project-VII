import numpy as np
from DDPG_Environment import SA_DDPG_Env, UEs, OUNoise
from DDPG_Agent import Agent
from tools import plotter, fungsi as fn



#DDPG Properties & Hyperparameters
sampling = 10
episode = 250
max_steps = 100

nn_actor = np.array([[512],[256]])
nn_critic = np.array([[512],[256]])
alpha, beta = 1e-3, 2e-3
gamma = 0.9
tau = 1e-4
batch_size = 32
min_sigma = 0.001
max_sigma = 0.1
decay_period = 100


#System Model
N = 3
M = 10
K_set = [10,20,30,40] #set of reflecting element of IRS
B = 1e6 #Bandwidth 1 Mhz
noise = fn.db2lin(-134)/1000 
b0 = -30
b1 = 4
PkPc = 4 #Pk+Pc watt
k1 = 2
k2 = 2.2
power_properties = np.array([[0],[5]]) 
phase_properties = np.array([[0],[2*np.pi]])
x_uav = np.array((0,200,400),dtype=np.float32)
y_uav = np.array((0,300,0),dtype=np.float32)
z_uav = np.array((200,200,200),dtype=np.float32)
irs_loc = np.array((500,500,30),dtype=np.float32)
x_ues = np.array((0,200,400),dtype=np.int_)
y_ues = np.array((0,300,0),dtype=np.int_)
coverage = 500 #radius
UEs = UEs(N,M,x_ues,y_ues,coverage)
UEs_loc = UEs.point

# UEs.plot_location()

plotter1 = plotter(episode,sampling,K_set) 
rewards_temp = np.zeros((episode,max_steps),dtype=np.float32)

for cntr in range(len(K_set)):
    K = K_set[cntr]
    
    for sample_ in range(sampling):         
        env = SA_DDPG_Env(N,M,K,B,noise,b0,b1,k1,k2,PkPc,power_properties,phase_properties,
                         x_uav,y_uav,z_uav,irs_loc,UEs_loc)
        agent = Agent(input_dims=(2*M*N), env=env,
                n_actions=(N+K),nn_actor=nn_actor, nn_critic=nn_critic,gamma=gamma,alpha=alpha,
                beta=beta,tau=tau,batch_size=batch_size)
        OUnoise = OUNoise(env=env,max_sigma=max_sigma, min_sigma=min_sigma, decay_period=decay_period)
        
        for i0 in range(episode):        
            state = agent.env.reset_state()
            OUnoise.reset()
            
            
            for i in range(max_steps):
                action = agent.choose_action(state)
                action = OUnoise.get_action(action,i)
                reward, new_state, throughput= env.step(action)
                agent.remember(state, action, reward, new_state)
                agent.learn()
                state = new_state
                
                rewards_temp[i0,i] = reward 
            avg_score = np.mean(rewards_temp[i0,-100:])    
            plotter1.record(avg_score,i0,sample_,cntr)
            
            print('K =',K,'Sample',sample_+1,'Episode',i0+1, 'reward %.1f' % (reward), 'avg score %.1f' % avg_score)
                       
        print('\nK =',K,' Sampling ',str(sample_+1),' Done!!!--- \n')
    
plotter1.plot(grid=np.int_(episode))
plotter1.plot_result()
