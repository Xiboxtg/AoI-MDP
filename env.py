import sys
import numpy as np
import time
import math
import copy
from gym import spaces
from gym.utils import seeding
from dist_ang_estimate import dist_ang_estimate
import scipy.io as io

if sys.version_info.major == 2: # py2
    import Tkinter as tk  
else:
    import tkinter as tk  # GUI

WIDTH = 3  # The width of the map.
HEIGHT = 3  # The height of the map.
UNIT = 40  # Pixel value.
LDA = [3., 5., 8., 12.]  # Assuming there are four types of sensors with varying priorities.
max_LDA = max(LDA)
C = 5000  # Packet capacity.
P_u = 3e-2 # Sensor power.
H = 10

N_S_ = 65  # Number of sensors.
V = 0.6  # Maximum speed of AUV.
S = 63 # Cross-section area.
b_S_ = np.random.randint(0, 1000, N_S_)  # Initialize the current data cache of the sensor.

np.random.seed(1)

def calcRate(f,b,d,numb=0):
    f1 = (f-b/2) if numb == 0 else (f+b/2)

    lgNt = 17 - 30*math.log10(f1) # Turbulent noise.
    lgNs = 40 + 26*math.log10(f1) - 60*math.log10(f+0.03) # Ship noise, assuming s=0.5.
    lgNw = 50 + 20*math.log10(f1) - 40*math.log10(f+0.4) # Sea surface noise, assuming w=0
    lgNth = -15 + 20*math.log10(f1) # Thermal noise

    NL = 10 * math.log10(1000*b*(10**(lgNt/10)+10**(lgNs/10)+10**(lgNw/10)+10**(lgNth/10)))

    alpha = 0.11*((f1**2)/(1+f1**2)) + 44*((f1**2)/(4100+f1**2)) + (2.75e-4)*(f1**2) + 0.003
    TL = 15 * math.log10(d) + alpha * (0.001 * d)

    SL = 10*math.log10(P_u) + 170.77
    #Calculate data collection rate.
    R = 0.001 * b * math.log(1+10**(SL-TL-NL),2)
    return R

def constrain(val,min,max):
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val


class UAV(tk.Tk, object):
    def __init__(self, R_dc=16.):
        super(UAV, self).__init__()
        self.N_POI = N_S_
        #Initialize VoI related variables.
        self.alpha = 0.7
        self.beta = 10  # VoI scaling factor.
        self.E_i = np.random.uniform(0, 50, self.N_POI)
        self.V_i = np.zeros(self.N_POI)
        self.Access_node = np.zeros(self.N_POI)# Record the index of all visited nodes.
        self.coll_time = 0#Record the data collection time of all nodes.
        self.t_fly = 0#Record sailing time.

        self.dis = np.zeros(self.N_POI)
        self.N_UAV = 2

        self.max_speed = V
        self.H = 10.
        self.X_min = 0
        self.Y_min = 0
        self.X_max = (WIDTH) * UNIT
        self.Y_max = (HEIGHT) * UNIT  # Map boundary.
        self.R_dc = R_dc  # Horizontal coverage distance.
        self.sdc = math.sqrt(pow(self.R_dc, 2) + pow(self.H, 2))  # Maximum service distance.
        self.f = 20 # khz
        self.b = 1

        # Load turbulence data.
#        turb = io.loadmat('turb.mat')
#        self.uu = turb['uu']
#        self.vv = turb['vv']

        self.FX = 0
        self.crash = 0
        self.SoPcenter = np.random.randint(10, 110, size=[self.N_POI, 2]) # Randomly initialize position.
        self.action_space = spaces.Box(low=np.array([0., -1.]), high=np.array([1., 1.]),
                                       dtype=np.float32)#Define the range of actions that AUVs can perform.
        self.state_dim = 8
        self.state1 = np.zeros(self.state_dim)
        self.state2 = np.zeros(self.state_dim)
        self.xy = np.zeros((self.N_UAV, 2))
        self.AutoUAV = []
        self.Aim = []

        # 传感器初始化
        CoLDA = np.random.randint(0, len(LDA), self.N_POI)
        self.lda = [LDA[CoLDA[i]] for i in range(self.N_POI)]
        self.b_S = np.random.randint(0., 1000., self.N_POI).astype(np.float32)
        self.Fully_buffer = C
        self.N_Data_overflow = 0  # Data overflow count.
        self.H_Data_overflow1 = 0  # Used for data overflow calculation in hovering situations.
        self.H_Data_overflow2 = 0
        self.update_VoI()

        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)]) # Prioritization

        self.idx_target = np.argsort(self.Q)[-self.N_UAV:]

        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.title('MAP')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self.build_maze()
        self.dist_ang_estimate = dist_ang_estimate()

    def update_VoI(self):

        for i in range(self.N_POI):
            if self.Access_node[i] == 0:
                self.V_i[i] = 0
            else:
                t = self.coll_time + self.t_fly
                self.V_i[i] = self.alpha * self.E_i[i] + (1 - self.alpha) * self.E_i[i] * np.exp(
                    -t / self.beta)

    def build_maze(self):

        self.canvas = tk.Canvas(self, bg='white', width=WIDTH * UNIT, height=HEIGHT * UNIT)

        for i in range(self.N_POI):

            if self.lda[i] == LDA[0]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 3, self.SoPcenter[i][1] - 3,
                    self.SoPcenter[i][0] + 3, self.SoPcenter[i][1] + 3,
                    fill='pink')
            elif self.lda[i] == LDA[1]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 3, self.SoPcenter[i][1] - 3,
                    self.SoPcenter[i][0] + 3, self.SoPcenter[i][1] + 3,
                    fill='blue')
            elif self.lda[i] == LDA[2]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 3, self.SoPcenter[i][1] - 3,
                    self.SoPcenter[i][0] + 3, self.SoPcenter[i][1] + 3,
                    fill='green')
            elif self.lda[i] == LDA[3]:
                self.canvas.create_oval(
                    self.SoPcenter[i][0] - 3, self.SoPcenter[i][1] - 3,
                    self.SoPcenter[i][0] + 3, self.SoPcenter[i][1] + 3,
                    fill='red')
        # Create drones and ensure mutual spacing.
        while True:
            self.xy = np.random.randint(20, 100, size=[self.N_UAV, 2]).astype(float)
            if np.linalg.norm(self.xy[0]-self.xy[1]) > 15:
                break 

        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - 6, self.xy[i][1] - 6,
                self.xy[i][0] + 6, self.xy[i][1] + 6,
                fill='yellow')
            self.AutoUAV.append(L_UAV) 
        
        pxy = self.SoPcenter[np.argsort(self.Q)[-self.N_UAV:]]
        for i in range(self.N_UAV):
            L_AIM = self.canvas.create_rectangle(
            pxy[i][0] - 5, pxy[i][1] - 5,
            pxy[i][0] + 5, pxy[i][1] + 5,
            fill='red')
            self.Aim.append(L_AIM)
        self.canvas.pack()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.render()
        for i in range(self.N_UAV):
            self.canvas.delete(self.AutoUAV[i])
        self.AutoUAV = []

        for i in range(len(self.Aim)):
            self.canvas.delete(self.Aim[i])
        
        # Re randomly initialize the drone positions and ensure that they are spaced apart from each other.
        while True:
            #np.random.seed(2)
            np.random.seed(int(time.time()))
            self.xy = np.random.randint(20, 100, size=[self.N_UAV, 2]).astype(float)
            np.random.seed(1)
            if np.linalg.norm(self.xy[0]-self.xy[1]) > 15:
                break
        for i in range(self.N_UAV):
            L_UAV = self.canvas.create_oval(
                self.xy[i][0] - 6, self.xy[i][1] - 6,
                self.xy[i][0] + 6, self.xy[i][1] + 6,
                fill='yellow')
            self.AutoUAV.append(L_UAV)
        self.FX = 0.  

        self.b_S = np.random.randint(0, 1000, self.N_POI)  # Initialize the current data cache of the sensor.
        self.b_S = np.asarray(self.b_S, dtype=np.float32)
        self.N_Data_overflow = 0  # Data overflow count.
        self.Q = np.array([self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)])
        self.idx_target = np.argsort(self.Q)[-self.N_UAV:]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.pxy = self.SoPcenter[self.idx_target]

        for i in range(self.N_UAV):
            L_AIM = self.canvas.create_rectangle(
            self.pxy[i][0] - 5, self.pxy[i][1] - 5,
            self.pxy[i][0] + 5, self.pxy[i][1] + 5,
            fill='red')
            self.Aim.append(L_AIM)


        self.state1 = np.concatenate(((self.xy[1]-self.xy[0]).flatten()/120.,(self.pxy[0] - self.xy[0]).flatten() / 120., self.xy[0].flatten() / 120., [0., 0.]))
        self.state2 = np.concatenate(((self.xy[0]-self.xy[1]).flatten()/120.,(self.pxy[1] - self.xy[1]).flatten() / 120., self.xy[1].flatten() / 120., [0., 0.]))
        return self.state1,self.state2

    def step_move(self,action,action2,numb=0,hover=False,turb=False):
        sum_VoI = np.sum(self.V_i)

        detX = (action[0] * (self.max_speed) + 1.3)* math.cos(action[1] * math.pi)#Incremental displacement in the x direction.
        detY = (action[0] * (self.max_speed) + 1.3) * math.sin(action[1] * math.pi)

        detX2 = (action2[0] * (self.max_speed) + 1.3) * math.cos(action2[1] * math.pi)
        detY2 = (action2[0] * (self.max_speed) + 1.3) * math.sin(action2[1] * math.pi)
        V = math.sqrt(pow(detX, 2) + pow(detY, 2))

        if turb == True:
            # Find the index of the current location.
            tx1 = constrain(int(round((self.xy[0,0] * 40) / 120, 0)) - 1,0,39)
            tx2 = constrain(int(round((self.xy[1,0] * 40) / 120, 0)) - 1,0,39)
            ty1 = constrain(int(round((self.xy[0,1] * 40) / 120, 0)) - 1,0,39)
            ty2 = constrain(int(round((self.xy[1,1] * 40) / 120, 0)) - 1,0,39)


#            if numb == 0:
#                detX += self.uu[ty1,tx1]
#                detY += self.vv[ty1,tx1]
#            else:
#                detX += self.uu[ty2,tx2]
#                detY += self.vv[ty2,tx2]
#            detX2 += self.uu[ty2,tx2]
#            detY2 += self.vv[ty2,tx2]
        state_ = np.zeros(self.state_dim)
        xy_ = copy.deepcopy(self.xy[numb])
        Flag = False

        if hover == True:
            detX = 0; detY = 0
        else:
            if numb == 0:
                self.H_Data_overflow1 = 0
            elif numb == 1:
                self.H_Data_overflow2 = 0
        xy_[0] += detX
        xy_[1] += detY

        if xy_[0] >= self.X_min and xy_[0] <= self.X_max:
            if xy_[1] >= self.Y_min and xy_[1] <= self.Y_max:
                self.FX = 0.
                Flag = True
            else:
                xy_[0] -= detX 
                xy_[1] -= detY
                self.FX = 1.
        else:
            xy_[0] -= detX
            xy_[1] -= detY
            self.FX = 1.

        if Flag and (hover == False):
            self.t_fly += 1
            # Flight energy consumption, formula calculation.
            F = (0.7*S*(V**2))/2
            ec = (F * V) / (-0.081*(V**3)+0.215*(V**2)-0.01*V+0.541)
        else:
            ec = 90  # Hover power consumption.

        self.canvas.move(self.AutoUAV[numb],xy_[0] - self.xy[numb][0], xy_[1] - self.xy[numb][1])

        self.xy[numb] = xy_
        real_position = copy.deepcopy(self.xy[numb]) #Copy the real location.
        self.xy[numb], time_delay = self.dist_ang_estimate.position_estimate(real_position) #Location and time delay estimation.

        if numb == 0:

            self.N_Data_overflow = 0
            self.b_S += [np.random.poisson(self.lda[i]) for i in range(self.N_POI)]  # Sensor data cache update.
            for i in range(self.N_POI):  # Data overflow handling.
                if self.b_S[i] >= self.Fully_buffer:
                    self.N_Data_overflow += 1
                    self.H_Data_overflow1 += 1
                    self.H_Data_overflow2 += 1
                    self.b_S[i] = self.Fully_buffer
            self.updata = self.b_S[self.idx_target] / self.Fully_buffer
            state_[:2] = (self.xy[1]+np.array([detX2,detY2])-xy_).flatten() / 120.
            sd = np.linalg.norm(state_[:2]*120)
            if sd < 5:
                self.crash = 1
            else:
                self.crash = 0
        else:
            state_[:2] = (self.xy[0]-xy_).flatten() / 120.

        state_[2:4] = (self.pxy[numb] - xy_).flatten() / 120.
        state_[4:6] = xy_.flatten() / 120.  # Absolute position of AUV.
        state_[6] = self.FX / 800.
        state_[7] = self.N_Data_overflow / self.N_POI
        Done = False
        # Get reward.
        reward = -5*np.linalg.norm(state_[2:4]*120) * 60 - self.FX * 10 - 5*self.N_Data_overflow * 5
        # Inverse proportional function obstacle avoidance penalty.
        if np.linalg.norm(state_[:2]*120) < 12:
            reward -= 300 * (12 - np.linalg.norm(state_[:2]*120))

        data_rate = 0
        self.calc_dist(numb)
        if self.dis[self.idx_target[numb]] <= self.sdc and hover == False:
            self.update_VoI()
            #reward += 5*np.sum(self.V_i)
            Done = True
            reward += 5200

            data_rate = max(calcRate(self.f,self.b,self.dis[self.idx_target[numb]],0),
                            calcRate(self.f,self.b,self.dis[self.idx_target[numb]],1))
            self.coll_time = self.b_S[self.idx_target[numb]]/data_rate
            self.t_fly = 0
            self.Access_node[self.idx_target] = 1
            self.b_S[self.idx_target[numb]] = 0
        self.xy[numb] = real_position

        if numb == 0 :
            self.state1 = state_
        elif numb == 1:
            self.state2 = state_
        if hover == False:
            return state_,reward,Done,data_rate,ec,sum_VoI, time_delay, self.crash

    def calc_dist(self,numb=0,HH=True):#Calculate the distance between the AUV and the node.
        for i in range(self.N_POI):
            if HH == True:
                self.dis[i] = math.sqrt(
                    pow(self.SoPcenter[i][0] - self.xy[numb][0], 2) + pow(self.SoPcenter[i][1] - self.xy[numb][1], 2) + pow(self.H, 2))
            else:
                self.dis[i] = math.sqrt(
                    pow(self.SoPcenter[i][0] - self.xy[numb][0], 2) + pow(self.SoPcenter[i][1] - self.xy[numb][1], 2))
            
    def CHOOSE_AIM(self,numb=0,lamda=0.):#Select target points based on priority.
        self.canvas.delete(self.Aim[numb])
        del self.Aim[numb]
        self.calc_dist(numb=numb,HH=False)

        Q = np.array([self.lda[i] * self.b_S[i] / self.Fully_buffer - lamda * self.dis[i] for i in range(self.N_POI)])
        idx_target = np.argsort(Q)[-self.N_UAV:]

        inter = np.intersect1d(idx_target,self.idx_target)

        if len(inter) == 0:
            self.idx_target[numb] = idx_target[0]       
        elif len(inter) < len(self.idx_target):

            diff = np.setdiff1d(idx_target,inter)
            self.idx_target[numb] = diff[0]
        else:

            idx_target = np.argsort(self.Q)[-(self.N_UAV+1):]
            self.idx_target[numb] = idx_target[0]

        self.pxy = self.SoPcenter[self.idx_target]
        L_AIM = self.canvas.create_rectangle(
            self.pxy[numb][0] - 5, self.pxy[numb][1] - 5,
            self.pxy[numb][0] + 5, self.pxy[numb][1] + 5,
            fill='red')
        self.Aim.insert(numb,L_AIM)

        # Get state.
        if numb == 0:
            state_ = self.state1; ofd = self.H_Data_overflow1
        elif numb == 1:
            state_ = self.state2; ofd = self.H_Data_overflow2
        state_[2:4] = (self.pxy[numb] - self.xy[numb][0]).flatten() / 120.
        state_[7] = ofd / self.N_POI
        self.render()
        return state_
        
    def render(self, t=0.0001):
        time.sleep(t)
        self.update()    