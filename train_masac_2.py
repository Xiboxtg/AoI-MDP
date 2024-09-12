# ---------------------
#Env: pytorch 1.13(cuda 11.7) + python 3.9
# ---------------------
#AoI-MDP

import math
import os
import time
from env import UAV
#from ddpg import AGENT
from masac import store_transition,SAC,is_training,store_data
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import pickle
parser = argparse.ArgumentParser(description='Train the SAC model.')
parser.add_argument('--is_train', type=int, default=1, metavar='train(1) or eval(0)',
                    help='train model of evaluate the trained model')
# TRAINING
parser.add_argument('--gamma', type=float, default=0.9, metavar='discount rate',
                    help='The discount rate of long-term returns')
parser.add_argument('--mem_size', type=int, default=8000, metavar='memorize size',
                    help='max size of the replay memory')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch size',
                    help='batch size')
parser.add_argument('--lr_actor', type=float, default=0.001, metavar='learning rate of actor',
                    help='learning rate of actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='learning rate of critic',
                    help='learning rate of critic network')
parser.add_argument('--replace_tau', type=float, default=0.001, metavar='replace_tau',
                    help='soft replace_tau')
parser.add_argument('--episode_num', type=int, default=400, metavar='episode number',
                    help='number of episodes for training')
parser.add_argument('--Num_episode_plot', type=int, default=10, metavar='plot freq',
                    help='frequent of episodes to plot')
parser.add_argument('--save_model_freq', type=int, default=20, metavar='save freq',
                    help='frequent to save network parameters')
parser.add_argument('--model', type=str, default='P_moddpg', metavar='save path',
                    help='the save path of the train model')
parser.add_argument('--R_dc', type=float, default=9., metavar='R_DC',
                    help='the radius of data collection')
parser.add_argument('--R_eh', type=float, default=30., metavar='R_EH',
                    help='the radius of energy harvesting')
parser.add_argument('--w_dc', type=float, default=200., metavar='W_DC',
                    help='the weight of data collection')
parser.add_argument('--w_eh', type=float, default=120., metavar='W_EH',
                    help='the weight of energy harvesting')
parser.add_argument('--w_ec', type=float, default=16., metavar='W_EC',
                    help='the weight of energy consumption')

args = parser.parse_args()

#####################  set the save path  ####################
model_path = '/{}/{}/'.format(args.model, 'models_m2')
path = os.getcwd() + model_path
if not os.path.exists(path):
    os.makedirs(path)
logs_path = '/{}/{}/'.format(args.model, 'logs_m2')
path = os.getcwd() + logs_path
if not os.path.exists(path):
    os.makedirs(path)
figs_path = '/{}/{}/'.format(args.model, 'figs_m2')
path = os.getcwd() + figs_path
if not os.path.exists(path):
    os.makedirs(path)


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
Q = 2

#def get_current_time():
#    return time.time()

def simulate_network_delay():
    delay = random.randint(0, 4)
#    time.sleep(delay)
    return delay

def delay_transformer(time_delay):
    if time_delay < 0.05:
        tdelay = 0
    if time_delay >= 0.05 and time_delay < 0.06:
        tdelay = 1
    if time_delay >= 0.06 and time_delay < 0.07:
        tdelay = 2
    if time_delay >= 0.07 and time_delay < 0.08:
        tdelay = 3
    if time_delay >=0.08:
        tdelay = 4
    return tdelay

#def calculate_aoi(current_time, state_time):
#    return current_time - state_time

def train():
    for ep in range(args.episode_num):
        s1,s2 = env.reset()
        #current_time = get_current_time()
        #s1_time, s2_time = current_time, current_time
        ep_reward = 0#Cumulative reward

        idu = 0
        N_DO = 0
        DQ = 0
        FX1 = 0
        FX2 = 0
        sum_rate = 0
        Ec1 = 0  # flying energy consumption
        Ec2 = 0
        TD_error1 = 0
        TD_error2 = 0
        A_loss1 = 0
        A_loss2 = 0 
        Ht1 = 0  # howering time
        Ht2 = 0 
        Ft = 0  # flying time
        update_network1 = 0
        update_network2 = 0
        crash = 0
        mode1:int = 0
        mode2:int = 0
        ht1 = 0
        ht2 = 0
        iffirst1 = 1
        iffirst2 = 1
        ifsample = 0
        z1 = []
        z2 = []
        z1.append(0)
        z2.append(0)
        zt1 = 1
        zt2 = 1
        ydelay1 = []
        ydelay2 = []
        aoi1 = 0
        aoi2 = 0
        while True:
            ft = 1
            if_is_sample = 0
            if zt1 > z1[-1]:
                if iffirst1 == 1:
                    act1, z1_1 = agent1.select_action(s1)
                    z1_1 = z1_1.item()
                    z1.append(math.floor(z1_1))
                    #act2, z2 = agent2.select_action(s2)
                    act1[0] = 0.5 * (act1[0] + 1)
                    #act2[0] = 0.5 * (act2[0] + 1)
                    #act2_ = act2
                    iffirst1 = 0
                    ifsample = 1
                    #zt = 0
                    ydelay1.append(0)
                    #delay0 = simulate_network_delay()
                    #ydelay1.append(delay0)
                    zt1 = 0
                if ifsample == 0:
                    #delay1 = simulate_network_delay()
                    #delay1_2 = simulate_network_delay()
                    #ydelay1.append(delay1)
                    #ydelay1.append(delay1_2)
                    act1, z1_1 = agent1.select_action(s1)
                    z1_1 = z1_1.item()
                    act1[0] = 0.5 * (act1[0] + 1)
                    z1.append(math.floor(z1_1))
                    zt1 = 0
                    s1_, r1, done1, dr1, ec1, sumV, time_delay1, cs = env.step_move(act1, act2_, numb=0, hover=False)
                    ydelay1.append(delay_transformer(time_delay1))
                    for i in range(ydelay1[-1]):
                        Ft += 1
                        if (is_training(numb=1)) and (Ft % 2 == 0):
                            agent1.update_parameters(256, numb=1)

                            update_network1 += 1
                            TD_error1 += agent1.q_loss.detach().cpu().numpy()
                            A_loss1 += agent1.policy_loss.detach().cpu().numpy()
                    if mode1 == 0:
                        #s1_, r1, done1, dr1, ec1, sumV, time_delay1, cs = env.step_move(act1, act2_, numb=0, hover=False)
                        # simulate_network_delay()
                        aoi1 += (2*ydelay1[-2] + ydelay1[-1] + z1[-1]) * (z1[-1] + ydelay1[-1])/2  #sum-AoI
                        aoi1_ave = aoi1/(sum(z1)+sum(ydelay1)+0.001) #time-averaged AoI
                        #ydelay1.append(delay_transformer(time_delay1))
                        #aoi1_ave = (ydelay1[-1] + z1[-1] * z1[-1]) / (2 * (ydelay1[-1] + z1[-1])) + ydelay1[-1]
                        crash += cs
                        r1 += (args.w_dc * dr1 - args.w_ec * ec1)
                        r1 -= 1000 * aoi1_ave

                        r1 /= 20000
                        ep_reward += r1

                        store_transition(s1, act1, r1, s1_, numb=1)

                        s1 = s1_

                        Ec1 += ec1

                        if done1 == True:

                            idu += 1
                            ht = Q * env.updata[0] / dr1
                            ht1 = ht

                            mode1 += math.ceil(ht)

                            sum_rate += dr1

                    else:
                        env.step_move(act1, act2_, numb=0, hover=True)
                        mode1 -= 1
                        Ht1 += 1

                        if mode1 == 0:

                            Ht1 -= (math.ceil(ht1) - ht1)
                            s1 = env.CHOOSE_AIM(numb=0)
                    FX1 += env.FX



                    N_DO += env.N_Data_overflow
                    DQ += sum(env.b_S / env.Fully_buffer)
            if zt2 > z2[-1]:
                if iffirst2 == 1:
                    #act1, z1 = agent1.select_action(s1)
                    act2, z2_1 = agent2.select_action(s2)
                    z2_1 = z2_1.item()
                    z2.append(math.floor(z2_1))
                    #act1[0] = 0.5 * (act1[0] + 1)
                    act2[0] = 0.5 * (act2[0] + 1)
                    act2_ = act2
                    iffirst2 = 0
                    ifsample = 1
                    #zt = 0
                    ydelay2.append(0)
                    #delay0_2 = simulate_network_delay()
                    #ydelay2.append(delay0_2)
                    zt2 = 0
                if ifsample == 0:
                    #delay2 = simulate_network_delay()
                    #delay2_2 = simulate_network_delay()
                    #ydelay2.append(delay2)
                    #ydelay2.append(delay2_2)
                    act2, z2_1 = agent2.select_action(s2)
                    z2_1 = z2_1.item()
                    act2[0] = 0.5 * (act2[0] + 1)
                    act2_ = act2
                    z2.append(math.floor(z2_1))
                    zt2 = 0
                    s2_, r2, done2, dr2, ec2, sumV, time_delay2, cs = env.step_move(act2, act1, numb=1, hover=False)
                    ydelay2.append(delay_transformer(time_delay2))
                    for i in range(ydelay2[-1]):
                        Ft += 1

                        if (is_training(numb=2)) and (Ft % 2 == 0):
                            agent2.update_parameters(256, numb=2)

                            update_network2 += 1
                            TD_error2 += agent2.q_loss.detach().cpu().numpy()
                            A_loss2 += agent2.policy_loss.detach().cpu().numpy()
                    if mode2 != 0:
                        act2_ = np.array([0, 0])

                    if mode2 == 0:
                        #s2_, r2, done2, dr2, ec2, sumV, time_delay2, cs = env.step_move(act2, act1, numb=1, hover=False)
                        aoi2 += (2*ydelay2[-2] + ydelay2[-1] + z2[-1]) * (z2[-1] + ydelay2[-1])/2  # sum-AoI
                        aoi2_ave = aoi2 / (sum(z2) + sum(ydelay2)+0.001)  # time-averaged AoI
                        #ydelay2.append(delay_transformer(time_delay2))
                        #aoi2_ave = (ydelay2[-1] + z2[-1] * z2[-1]) / (2 * (ydelay2[-1] + z2[-1])) + ydelay2[-1]
                        # simulate_network_delay()
                        #aoi2_ = calculate_aoi(get_current_time(), s2_time_)
                        # crash += cs
                        r2 += (args.w_dc * dr2 - args.w_ec * ec2)
                        r2 -= 1000 * aoi2_ave
                        r2 /= 20000
                        ep_reward += r2

                        store_transition(s2, act2, r2, s2_, numb=2)

                        s2 = s2_

                        Ec2 += ec2

                        if done2 == True:

                            idu += 1
                            ht = Q * env.updata[1] / dr2
                            ht2 = ht

                            mode2 += math.ceil(ht)

                            sum_rate += dr2
                    else:
                        env.step_move(act2, act1, numb=1, hover=True)
                        mode2 -= 1

                        Ht2 += 1

                        if mode2 == 0:

                            Ht2 -= (math.ceil(ht2) - ht2)
                            s2 = env.CHOOSE_AIM(numb=1)
                    FX2 += env.FX


                    N_DO += env.N_Data_overflow
                    DQ += sum(env.b_S / env.Fully_buffer)




            #act1[0] = 0.5 * (act1[0] + 1)
            #act2[0] = 0.5 * (act2[0] + 1)
            #act2_ = act2
            if ifsample == 1:
                ifsample = 0
                if_is_sample = 1
                if mode2 != 0:
                    act2_ = np.array([0, 0])
                if mode1 == 0:
                    s1_, r1, done1, dr1, ec1, sumV, time_delay1, cs = env.step_move(act1, act2_, numb=0, hover=False)
                    zt1 += 1
                    # simulate_network_delay()
                    #aoi1 += (2 * ydelay1[-1] + z1[-1]) * z1[-1] / 2
                    #aoi1_ave = aoi1 / (sum(z1) + sum(ydelay1) + 0.001)
                    #ydelay1.append(delay_transformer(time_delay1))
                    #aoi1_ave = (ydelay1[-1] + z1[-1] * z1[-1]) / (2 * (ydelay1[-1] + z1[-1])) + ydelay1[-1]
                    crash += cs
                    r1 += (args.w_dc * dr1 - args.w_ec * ec1)
                    #r1 -= 500 * aoi1_ave

                    r1 /= 20000
                    ep_reward += r1

                    store_transition(s1, act1, r1, s1_, numb=1)

                    s1 = s1_

                    Ec1 += ec1

                    if done1 == True:

                        idu += 1
                        ht = Q * env.updata[0] / dr1
                        ht1 = ht

                        mode1 += math.ceil(ht)

                        sum_rate += dr1

                else:
                    env.step_move(act1, act2_, numb=0, hover=True)
                    mode1 -= 1
                    Ht1 += 1

                    if mode1 == 0:

                        Ht1 -= (math.ceil(ht1) - ht1)
                        s1 = env.CHOOSE_AIM(numb=0)
                FX1 += env.FX

                if mode2 == 0:
                    s2_, r2, done2, dr2, ec2, sumV, time_delay2, cs = env.step_move(act2, act1, numb=1, hover=False)
                    zt2 += 1
                    #aoi2 += (2 * ydelay2[-1] + z2[-1]) * z2[-1] / 2
                    #aoi2_ave = aoi2 / (sum(z2) + sum(ydelay2) + 0.001)
                    #ydelay2.append(delay_transformer(time_delay2))
                    #aoi2_ave = (ydelay2[-1] + z2[-1] * z2[-1]) / (2 * (ydelay2[-1] + z2[-1])) + ydelay2[-1]
                    # simulate_network_delay()
                    # aoi2_ = calculate_aoi(get_current_time(), s2_time_)
                    # crash += cs
                    r2 += (args.w_dc * dr2 - args.w_ec * ec2)
                    #r2 -= 500 * aoi2_ave
                    r2 /= 20000
                    ep_reward += r2

                    store_transition(s2, act2, r2, s2_, numb=2)

                    s2 = s2_

                    Ec2 += ec2

                    if done2 == True:

                        idu += 1
                        ht = Q * env.updata[1] / dr2
                        ht2 = ht

                        mode2 += math.ceil(ht)

                        sum_rate += dr2
                else:
                    env.step_move(act2, act1, numb=1, hover=True)
                    mode2 -= 1

                    Ht2 += 1

                    if mode2 == 0:

                        Ht2 -= (math.ceil(ht2) - ht2)
                        s2 = env.CHOOSE_AIM(numb=1)
                FX2 += env.FX


                N_DO += env.N_Data_overflow
                DQ += sum(env.b_S / env.Fully_buffer)
                Ft += 1
                if (is_training(numb=1)) and (Ft % 2 == 0):
                    agent1.update_parameters(256, numb=1)

                    update_network1 += 1
                    TD_error1 += agent1.q_loss.detach().cpu().numpy()
                    A_loss1 += agent1.policy_loss.detach().cpu().numpy()

                if (is_training(numb=2)) and (Ft % 2 == 0):
                    agent2.update_parameters(256, numb=2)

                    update_network2 += 1
                    TD_error2 += agent2.q_loss.detach().cpu().numpy()
                    A_loss2 += agent2.policy_loss.detach().cpu().numpy()
            if if_is_sample == 0:
                # if mode2 != 0:
                #     act2_ = np.array([0, 0])
                #if mode1 == 0:
                    #s1_, r1, done1, dr1, ec1, sumV, cs = env.step_move(act1, act2_, numb=0, hover=False)
                zt1 += 1
                    # simulate_network_delay()
                    # aoi1 += (2 * ydelay1[-1] + z1[-1]) * z1[-1] / 2
                    # aoi1_ave = aoi1 / (sum(z1) + sum(ydelay1))
                    # crash += cs
                #     r1 += (args.w_dc * dr1 - args.w_ec * ec1)
                #     r1 -= 10 * aoi1_ave
                #
                #     r1 /= 10000
                #     ep_reward += r1
                #
                #     store_transition(s1, act1, r1, s1_, numb=1)
                #
                #     s1 = s1_
                #
                #     Ec1 += ec1
                #
                #     if done1 == True:
                #
                #         idu += 1
                #         ht = Q * env.updata[0] / dr1
                #         ht1 = ht
                #
                #         mode1 += math.ceil(ht)
                #
                #         sum_rate += dr1
                #
                # else:
                #     env.step_move(act1, act2_, numb=0, hover=True)
                #     mode1 -= 1
                #     Ht1 += 1
                #
                #     if mode1 == 0:
                #
                #         Ht1 -= (math.ceil(ht1) - ht1)
                #         s1 = env.CHOOSE_AIM(numb=0)
                # FX1 += env.FX
                #
                # if mode2 == 0:
                #     s2_, r2, done2, dr2, ec2, sumV, cs = env.step_move(act2, act1, numb=1, hover=False)
                zt2 += 1
                #     aoi2 += (2 * ydelay2[-1] + z2[-1]) * z2[-1] / 2  # 总AoI
                #     aoi2_ave = aoi2 / (sum(z2) + sum(ydelay2))
                #     # simulate_network_delay()
                #     # aoi2_ = calculate_aoi(get_current_time(), s2_time_)
                #     # crash += cs
                #     r2 += (args.w_dc * dr2 - args.w_ec * ec2)
                #     r2 -= 10 * aoi2_ave
                #     r2 /= 10000
                #     ep_reward += r2
                #
                #
                #     store_transition(s2, act2, r2, s2_, numb=2)
                #
                #     s2 = s2_
                #
                #     Ec2 += ec2
                #
                #     if done2 == True:
                #
                #         idu += 1
                #         ht = Q * env.updata[1] / dr2
                #         ht2 = ht
                #
                #         mode2 += math.ceil(ht)
                #
                #         sum_rate += dr2
                # else:
                #     env.step_move(act2, act1, numb=1, hover=True)
                #     mode2 -= 1
                #
                #     Ht2 += 1
                #
                #     if mode2 == 0:
                #
                #         Ht2 -= (math.ceil(ht2) - ht2)
                #         s2 = env.CHOOSE_AIM(numb=1)
                # FX2 += env.FX
                #
                #
                # N_DO += env.N_Data_overflow
                # DQ += sum(env.b_S / env.Fully_buffer)
                Ft += 1
                if (is_training(numb=1)) and (Ft % 2 == 0):
                    agent1.update_parameters(256, numb=1)

                    update_network1 += 1
                    TD_error1 += agent1.q_loss.detach().cpu().numpy()
                    A_loss1 += agent1.policy_loss.detach().cpu().numpy()

                if (is_training(numb=2)) and (Ft % 2 == 0):
                    agent2.update_parameters(256, numb=2)

                    update_network2 += 1
                    TD_error2 += agent2.q_loss.detach().cpu().numpy()
                    A_loss2 += agent2.policy_loss.detach().cpu().numpy()

                if Ft > 4000:
                    if update_network1 != 0:
                        TD_error1 /= update_network1
                        A_loss1 /= update_network1
                    if update_network2 != 0:
                        TD_error2 /= update_network2
                        A_loss2 /= update_network2

                    N_DO /= Ft
                    DQ /= Ft
                    DQ /= env.N_POI
                    FX1 /= (Ft - Ht1)
                    FX2 /= (Ft - Ht2)
                    Ec1 /= (0.5*(Ft - Ht1))
                    Ec2 /= (0.5*(Ft - Ht2))
                    print(
                        'TD_error1:%.2f | TD_error2:%.2f | A_loss1:%.2f |A_loss2:%.2f |ep_r:%i |L_data:%.2f |sum rate:%.2f |idu:%i |ec:%.2f |sumVoI:%.2f |N_D:%i |FX1:%.1f |FX2:%.1f |AOI1:%.2f |AoI2:%.2f |delay1:%.2f |delay2:%.2f |CS:%i' % (
                            TD_error1, TD_error2, A_loss1, A_loss2, ep_reward, DQ, sum_rate, idu, (Ec1 + Ec2) / 2, sumV,
                            N_DO, FX1, FX2, aoi1_ave, aoi2_ave, time_delay1, time_delay2, crash))
                    plot_x.append(ep)
                    plot_TD_error.append(TD_error1 + TD_error2)
                    plot_A_loss.append(A_loss1 + A_loss2)
                    plot_R.append(ep_reward)
                    plot_N_DO.append(N_DO)
                    plot_DQ.append(DQ)
                    plot_sr.append(sum_rate)
                    plot_aoi1.append(aoi1_ave)
                    plot_aoi2.append(aoi2_ave)
                    plot_VoI.append(sumV)
                    plot_Eh.append(crash)
                    plot_ehu.append(0)
                    plot_idu.append(idu)
                    plot_Ec.append((Ec1 + Ec2) / 2)
                    plot_HT.append((Ht1 + Ht2) / 2)
                    plot_FT.append(Ft - (Ht1 + Ht2) / 2)
                    break


        if ep % args.save_model_freq == 0 and ep != 0:
            agent1.save_models(model_path,ep)
            agent2.save_models(model_path,ep)
            # 保存变量
            with open('reward.pkl','wb') as f:
                pickle.dump(plot_R,f)
            with open('sum_rate.pkl','wb') as f:
                pickle.dump(plot_sr,f)
            with open('ec.pkl','wb') as f:
                pickle.dump(plot_Ec,f)
            with open('crash.pkl','wb') as f:
                pickle.dump(plot_Eh,f)
            with open('aoi1.pkl','wb') as f:
                pickle.dump(plot_aoi1,f)
            with open('aoi2.pkl','wb') as f:
                pickle.dump(plot_aoi2,f)
            with open('sumVoI.pkl','wb') as f:
                pickle.dump(plot_VoI,f)

            paint()
    store_data(numb=1)
    store_data(numb=2)






def paint():

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(labelsize=16)
    ax.grid(linestyle='-.')

    ax.plot(plot_x, plot_R)
    ax.set_xlabel('Number of training episodes', font1)
    ax.set_ylabel('Accumulated reward', font1)

    label1 = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]
    fig.tight_layout()

    plt.savefig('.{}{}'.format(figs_path, 'Accumulated_reward.jpg'))

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(plot_x, plot_aoi1)
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('AoI1', font1)
    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(plot_x, plot_aoi2)
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel('AoI2', font1)
    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'AoI.jpg'))
    #######################################1、loss##############################################

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')

    ax1.plot(plot_x, plot_A_loss)
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('loss of Actor', font1)
    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]


    ax2.plot(plot_x, plot_TD_error)
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel('td_error of critic', font1)
    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'loss.jpg'))
    ####################################################
    avg_ehu = [plot_ehu[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_ehu))]
    avg_eh = [plot_Eh[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_Eh))]
    eh_rate = [plot_Eh[i] / plot_HT[i] if plot_HT[i] != 0 else 0 for i in range(len(plot_Eh))]
    plot_r = [plot_sr[i] / plot_idu[i] if plot_idu[i] != 0 else 0 for i in range(len(plot_sr))]
    x = 0
    plot_sr1 = 0
    plot_r1 = 0
    plot_Eh1 = 0
    plot_Eh2 = 0
    plot_Eh3 = 0
    plot_Ec1 = 0
    plot_idu1 = 0
    plot_ehu1 = 0
    plot_ehu2 = 0
    plot_DQ1 = 0
    plot_N_DO1 = 0

    plot_x_avg = []
    plot_sr_avg = []
    plot_r_avg = []
    plot_Eh_avg = []
    plot_avg_Eh_avg = []
    plot_avg_Eh_rate = []
    plot_Ec_avg = []
    plot_idu_avg = []
    plot_ehu_avg = []
    plot_avg_ehu_avg = []
    plot_DQ_avg = []
    plot_N_DO_avg = []

    for i in range(1, len(plot_x)):
        x += i
        plot_sr1 += plot_sr[i]
        plot_r1 += plot_r[i]
        plot_Eh1 += plot_Eh[i]
        plot_Eh2 += avg_eh[i]
        plot_Eh3 += eh_rate[i]
        plot_Ec1 += plot_Ec[i]
        plot_idu1 += plot_idu[i]
        plot_ehu1 += plot_ehu[i]
        plot_ehu2 += avg_ehu[i]
        plot_DQ1 += plot_DQ[i]
        plot_N_DO1 += plot_N_DO[i]
        if i % args.Num_episode_plot == 0 and i != 0:
            plot_x_avg.append(x / args.Num_episode_plot)
            plot_sr_avg.append(plot_sr1 / args.Num_episode_plot)
            plot_r_avg.append(plot_r1 / args.Num_episode_plot)
            plot_Eh_avg.append(plot_Eh1 / args.Num_episode_plot)
            plot_avg_Eh_avg.append(plot_Eh2 / args.Num_episode_plot)
            plot_avg_Eh_rate.append(plot_Eh3 / args.Num_episode_plot)
            plot_Ec_avg.append(plot_Ec1 / args.Num_episode_plot)
            plot_idu_avg.append(plot_idu1 / args.Num_episode_plot)
            plot_ehu_avg.append(plot_ehu1 / args.Num_episode_plot)
            plot_avg_ehu_avg.append(plot_ehu2 / args.Num_episode_plot)
            plot_DQ_avg.append(plot_DQ1 / args.Num_episode_plot)
            plot_N_DO_avg.append(plot_N_DO1 / args.Num_episode_plot)

            x = 0
            plot_sr1 = 0
            plot_r1 = 0
            plot_Eh1 = 0
            plot_Eh2 = 0
            plot_Eh3 = 0
            plot_Ec1 = 0
            plot_idu1 = 0
            plot_ehu1 = 0
            plot_ehu2 = 0
            plot_DQ1 = 0
            plot_N_DO1 = 0

        #####################################################################

    # Fig 1:Average data rate(sum_rate/idu)/Total harvested energy(Ec)/Average flying energy consumption(Ec/Ft)
    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    ############################################main_result_1########################
    p1 = plt.figure(figsize=(24, 8))

    ax1 = p1.add_subplot(1, 3, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = p1.add_subplot(1, 3, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')
    ax3 = p1.add_subplot(1, 3, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_sr_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('sum data-rate (bits/Hz)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'Total harvested energy ($\mu$W)', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_Ec_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average flying energy consumption (W)', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'FIG_3.jpg'))
    ####################################################################################################################
    # Fig 2:Total number of DC devices(idu)/Average energy harvesting rate(Ec/Ht)/Average number of EH devices(ehu/idu)
    p1 = plt.figure(figsize=(22, 18))

    ax1 = p1.add_subplot(2, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = p1.add_subplot(2, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')
    ax3 = p1.add_subplot(2, 2, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')
    ax4 = p1.add_subplot(2, 2, 4)
    ax4.tick_params(labelsize=12)
    ax4.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_idu_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('Total number of DC devices', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_r_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel('Average data-rate (bits/Hz)', font1)
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_avg_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average number of EH devices', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]

    ax4.plot(plot_x_avg, plot_avg_Eh_rate, marker='*', markersize='10', linewidth='2')
    ax4.set_xlabel('Number of training episodes', font1)
    ax4.set_ylabel(r'Average energy harvesting rate ($\mu$W/s)', font1)

    label4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label4]

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'FIG_4.jpg'))
    ############################################data_rate/harvested energy########################
    ####################################fly energy consumption/total number of EH user########
    p1 = plt.figure(figsize=(28, 14))

    ax1 = p1.add_subplot(2, 4, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = p1.add_subplot(2, 4, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')
    ax3 = p1.add_subplot(2, 4, 3)
    ax3.tick_params(labelsize=12)
    ax3.grid(linestyle='-.')
    ax4 = p1.add_subplot(2, 4, 4)
    ax4.tick_params(labelsize=12)
    ax4.grid(linestyle='-.')
    ax5 = p1.add_subplot(2, 4, 5)
    ax5.tick_params(labelsize=12)
    ax5.grid(linestyle='-.')
    ax6 = p1.add_subplot(2, 4, 6)
    ax6.tick_params(labelsize=12)
    ax6.grid(linestyle='-.')
    ax7 = p1.add_subplot(2, 4, 7)
    ax7.tick_params(labelsize=12)
    ax7.grid(linestyle='-.')
    ax8 = p1.add_subplot(2, 4, 8)
    ax8.tick_params(labelsize=12)
    ax8.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_r_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('data rate (bits/Hz)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'Harvested energy ($\mu$W)', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    ax3.plot(plot_x_avg, plot_Ec_avg, marker='*', markersize='10', linewidth='2')
    ax3.set_xlabel('Number of training episodes', font1)
    ax3.set_ylabel('Average fly energy consumption (W)', font1)

    label3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label3]
    # ax3.legend(prop=font2)

    ax4.plot(plot_x_avg, plot_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax4.set_xlabel('Number of training episodes', font1)
    ax4.set_ylabel('Total number of EH user', font1)

    label4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label4]
    # ax4.legend(prop=font2)

    ax5.plot(plot_x_avg, plot_sr_avg, marker='*', markersize='10', linewidth='2')
    ax5.set_xlabel('Number of training episodes', font1)
    ax5.set_ylabel('sum rate (bits/Hz)', font1)

    label5 = ax5.get_xticklabels() + ax5.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label5]
    # ax5.legend(prop=font2)

    ax6.plot(plot_x_avg, plot_idu_avg, marker='*', markersize='10', linewidth='2')
    ax6.set_xlabel('Number of training episodes', font1)
    ax6.set_ylabel('Total number of ID user', font1)

    label6 = ax6.get_xticklabels() + ax6.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label6]
    # ax6.legend(prop=font2)

    ax7.plot(plot_x_avg, plot_avg_Eh_avg, marker='*', markersize='10', linewidth='2')
    ax7.set_xlabel('Number of training episodes', font1)
    ax7.set_ylabel(r'Average harvested energy ($\mu$W)', font1)

    label7 = ax7.get_xticklabels() + ax7.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label7]
    # ax7.legend(prop=font2)

    ax8.plot(plot_x_avg, plot_avg_ehu_avg, marker='*', markersize='10', linewidth='2')
    ax8.set_xlabel('Number of training episodes', font1)
    ax8.set_ylabel('Average number of EH user', font1)

    label8 = ax8.get_xticklabels() + ax8.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label8]
    # ax8.legend(prop=font2)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('.{}{}'.format(figs_path, 'sum_up.jpg'))
    plt.clf()
    #################################################################################################
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.tick_params(labelsize=12)
    ax1.grid(linestyle='-.')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.tick_params(labelsize=12)
    ax2.grid(linestyle='-.')

    ax1.plot(plot_x_avg, plot_DQ_avg, marker='*', markersize='10', linewidth='2')
    ax1.set_xlabel('Number of training episodes', font1)
    ax1.set_ylabel('Average data buffer length (%)', font1)

    label1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label1]

    ax2.plot(plot_x_avg, plot_N_DO_avg, marker='*', markersize='10', linewidth='2')
    ax2.set_xlabel('Number of training episodes', font1)
    ax2.set_ylabel(r'$N_d^{AVG}$', font1)

    label2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in label2]

    plt.savefig('.{}{}'.format(figs_path, 'system_performance.jpg'))
    plt.show
    ###################################################################################
    now_time = datetime.datetime.now()
    date = now_time.strftime('%Y-%m-%d %H_%M_%S')
    print('Running time: ', time.time() - t1)




if __name__ == '__main__':

    # Initialize environment.
    env = UAV(args.R_dc)
    s_dim = env.state_dim
    a_num = 2
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space
    agent1 = SAC(s_dim,a_dim,numb=1)
    #agent1.load_models('P_moddpg',100)
    agent2 = SAC(s_dim,a_dim,numb=2)
    #agent2.load_models('P_moddpg',100)
    # Train
    t1 = time.time()

    plot_x = []
    plot_R = []
    plot_N_DO = []
    plot_DQ = []
    plot_sr = []
    plot_Eh = []
    plot_Ec = []
    plot_ehu = []
    plot_idu = []
    plot_HT = []
    plot_FT = []
    plot_TD_error = []
    plot_A_loss = []
    plot_aoi1 = []
    plot_aoi2 = []
    plot_VoI = []
    file = open(os.path.join('.{}{}'.format(logs_path, 'log.txt')), 'w+')
    train()
    file.close()
