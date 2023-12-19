import numpy as np
import matplotlib.pyplot as plt


class fungsi:
    
    def db2lin(db):
        lin = np.power(10,(db/10))
        return lin

class plotter:
    def __init__(self, steps, sample,num_fig):
        self.steps = steps
        self.num_fig = len(num_fig)
        self.data_memory = np.zeros([np.int_(steps),np.int_(sample),np.int_(self.num_fig)])
        self.axis_memory = np.zeros(np.int_(steps))
        
        self.label = num_fig
        self.index = 0
        
    def record(self, rate, steps_th,sample_th,fig_th):
        self.data_memory[steps_th,sample_th,fig_th] = rate
        self.axis_memory[steps_th] = steps_th
            
    def plot(self, title="DDPG Single-Agent", ax="Episodes", ay="Rate", grid=1, smoother=1, fig=1):
        plotting = np.mean(self.data_memory,axis=1)
        plot_interval = np.int_(self.steps/grid)
        rate_set = np.zeros(grid)
        axis_set = np.zeros((grid),dtype=np.int_)
        if smoother < 0.1:
            smoother = 0.1
        smoother = np.log10(10*smoother) #To make logaritmic scale for ease to see as linear scale
        while plot_interval-(1-smoother)*plot_interval<1:
            smoother+=0.01
        
        plt.figure(fig)
        for i0 in range(self.num_fig):
                    
            for i in range(grid):
                rate_set[i]=np.sum(plotting[np.int_((i+(1-smoother))*plot_interval):(i+1)*plot_interval,i0])/plot_interval/smoother
                axis_set[i]=self.axis_memory[i*plot_interval]
            
            plt.plot(axis_set, rate_set, linewidth=1, label="K = "+str(self.label[i0]))
            plt.legend(loc='lower right', fontsize=10)
            plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1), np.max(axis_set)+5)
            plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                      np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
            plt.autoscale(enable=False, axis='x')
            
    
            plt.xlabel(ax)
            plt.ylabel(ay)
            plt.minorticks_on()
            plt.grid(b=True, which='major')
            plt.grid(b=True, which='minor',alpha=0.4)
            plt.suptitle(title, fontsize='x-large', fontweight='bold')
        plt.show() 
        
        
    def plot_result(self, title ="(DDPG) rate Vs Number of Reflecting Element on IRS",
                    ax="Number of Reflecting Element IRS", ay="Energy Efficiency (bits/J)"):
        
        
        plotting = np.mean(self.data_memory,axis=1)[-10:,:]
        plotting = np.mean(plotting,axis=0)
        axis_set = self.label
        
        plt.figure()
        plt.plot(axis_set, plotting, linewidth=1,marker="o")
        plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1),
                 np.max(axis_set)+(np.min(axis_set)*0.1))
        plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                  np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
        plt.autoscale(enable=False, axis='x')
                
        plt.xlabel(ax)
        plt.ylabel(ay)
        plt.grid(which='major', axis='both')
        plt.suptitle(title, fontsize='large', fontweight='bold')     
        
        plt.show()
                    