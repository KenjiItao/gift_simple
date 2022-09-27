import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
import random
from collections import Counter
from scipy.optimize import curve_fit



if int(sys.argv[1])==0:
    if not os.path.exists("./res"):
        os.mkdir("./res")
        os.mkdir("./figs")


def linear_fit(x, a, b):
    return a * x + b

def gini2(y):
    y.sort()
    n = len(y)
    nume = 0
    for i in range(n):
        nume = nume + (i+1)*y[i]

    deno = n*sum(y)
    return ((2*nume)/deno - (n+1)/(n))*(n/(n-1))

# class State:
#     def __init__(self, families):
#         self.families = families
#         self.network = np.ones([num_families, num_families])
#         # self.df = pd.DataFrame()

class Family:
    def __init__(self, family_id,  wealth):
        self.family_id = family_id
        self.wealth = wealth
        self.debt = []
        self.subordinates = []
        self.score = 0
        # self.lifetime = 0

def generation(families, network):
    family_ids = list(range(num_families))
    random.shuffle(family_ids)
    donation_ls = []
    for doner_id in family_ids:
        doner = families[doner_id]
        if doner.wealth > 0:
            recipient_id = random.choices(list(range(num_families)), weights = network[:, doner_id], k= 1)[0]
            if doner_id != recipient_id:
                recipient = families[recipient_id]
                recipient.debt.append([doner, doner.wealth * r])
                network[recipient_id, doner_id] += 1
                recipient.score += 1
                donation_ls.append([recipient_id, doner_id])

    for family in families:
        family.wealth += 1

    random.shuffle(family_ids)
    for family_id in family_ids:
        family = families[family_id]
        count = 0
        # random.shuffle(family.debt)
        for debt in family.debt:
            owner = debt[0]
            debt_wealth = debt[1]
            if family not in owner.subordinates and [family.family_id, owner.family_id] not in donation_ls:
                count += 1
                continue
            elif family.wealth >= debt_wealth:
                owner.wealth += debt_wealth
                if family in owner.subordinates:
                    owner.subordinates.remove(family)
                family.wealth -= debt_wealth
                network[owner.family_id, family_id] += 1
                owner.score += 1
                count += 1
            else:
                owner.wealth += family.wealth
                debt[1] -= family.wealth
                family.wealth = 0
                network[owner.family_id, family_id] += 1
                owner.score += 1
        family.debt = family.debt[count:]
        for debt in family.debt:
            owner = debt[0]
            if not family in owner.subordinates:
                owner.subordinates.append(family)

    # wealths = [family.wealth for family in families]
    # scores = np.sum(network, axis = 1) - num_families
    wealths, scores = [], []

    for family in families:
        # family.lifetime += 1
        if random.random() < 1 / l:
            wealths.append(family.wealth)
            # scores.append(family.score)
            scores.append(np.sum(network[family.family_id, :]) - num_families)
            family.wealth = 0.0
            family.score = 0
            # family.lifetime = 0
            family.debt = []
            family.subordinates = []
            network[family.family_id, :] = 1
            network[:, family.family_id] = 1

    return families, network, wealths, scores

np.sum(network, axis = 1) - num_families
[family.score for family in families]
sum(network[0])

scores
for family in families:
    print(family.wealth, len(family.subordinates))

def mean(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0



def main():
    families = [Family(i,  0.0) for i in range(num_families)]
    network = np.ones([num_families, num_families])

    # independent_duration_res, subordinate_duration_res, rich_duration_res = [], [], []
    wealth_ls, score_ls = [], []
    tot_wealth, tot_score = [], []
    iter = 0
    while iter < iteration:
        # print(iter)
        families, network, wealths, scores = generation(families, network)
        tot_wealth.append(sum(wealths))
        tot_score.append(sum(scores))
        iter += 1
        if iter >= iteration * 0.9:
            wealth_ls.extend(wealths)
            score_ls.extend(scores)


    # iter = 0
    # while iter < iteration:
    #     families, network, wealths, scores = generation(families, network)
    #     iter += 1
    #     wealth_ls.extend(wealths)
    #     score_ls.extend(scores)

    if iter == iteration and trial < 3:
        try:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            # ax.plot(np.arange(int(iteration * 0.9) - 1, iteration, 1), tot_wealth)
            ax.plot(tot_wealth)
            ax.set_xlabel("total wealth",fontsize=24)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_tot_wealth.pdf")
            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(tot_score)
            ax.set_xlabel("total score",fontsize=24)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_tot_score.pdf")
            plt.close('all')

            weights = np.ones(len(wealth_ls)) / len(wealth_ls)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            # ax.hist(wealth_ls, bins = 50, density = 1)
            ax.hist(wealth_ls, bins = 50, weights = weights)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log.pdf")
            plt.close('all')

            # wealth_ls = np.array(wealth_ls)
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ax.hist(wealth_ls[wealth_ls < 50], bins = 50, weights = weights)
            # ax.set_xlabel("wealth",fontsize=24)
            # ax.set_yscale('log')
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_wealth_log_exp.pdf")
            # plt.close('all')

            max_val = max(10, np.max(wealth_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(wealth_ls, bins = np.logspace(0, np.log10(np.max(wealth_ls)), 50), weights = weights)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, max_val + 10)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=22)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_log.pdf")
            plt.close('all')

            sx = sorted(wealth_ls)[::-1]
            N = len(wealth_ls)
            sy = [i/N for i in range(N)]

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_CDF.pdf")
            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_log_CDF.pdf")
            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls, bins = 50, weights = weights)
            ax.set_xlabel("score",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log.pdf")
            plt.close('all')

            # score_ls = np.array(score_ls)
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ax.hist(score_ls[score_ls < 50], bins = 50, weights = weights)
            # ax.set_xlabel("score",fontsize=24)
            # ax.set_yscale('log')
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_score_log_exp.pdf")
            # plt.close('all')

            max_val = max(1, np.max(score_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls, bins = np.logspace(0, np.log10(np.max(score_ls)), 50), weights = weights)
            ax.set_xlabel("score",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, max_val + 10)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=22)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_log.pdf")
            plt.close('all')

            sx = sorted(score_ls)[::-1]
            N = len(score_ls)
            sy = [i/N for i in range(N)]

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("score",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_CDF.pdf")
            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("score",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_log_CDF.pdf")
            plt.close('all')
        except:
            pass

    if True:
        wealth_exp_10pc, wealth_power_10pc, wealth_phase_10pc = np.nan, np.nan, np.nan
        score_exp_10pc, score_power_10pc, score_phase_10pc = np.nan, np.nan, np.nan
        try:
            wealth_sizes = np.array(wealth_ls)
            wealth_sizes.sort()
            sizes = wealth_sizes[::-1][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.1)]
            # wealths = np.log(sizes)
            wealths = sizes
            (val, bins)  = np.histogram(wealths)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp_10pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            wealth_power_10pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            wealth_phase_10pc = math.log(mse_exp / mse_power)

            sx = sorted(wealth_ls)[::-1]
            N = len(wealth_ls)
            sy = [i/N for i in range(N)]
        except:
            pass

        try:
            score_sizes = np.array(score_ls)
            score_sizes.sort()
            sizes = score_sizes[::-1][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.1)]
            # scores = np.log(sizes)
            scores = sizes
            (val, bins)  = np.histogram(scores)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            score_exp_10pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            score_power_10pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            score_phase_10pc = math.log(mse_exp / mse_power)

        except:
            pass

        wealth_exp_30pc, wealth_power_30pc, wealth_phase_30pc = np.nan, np.nan, np.nan
        score_exp_30pc, score_power_30pc, score_phase_30pc = np.nan, np.nan, np.nan
        try:
            wealth_sizes = np.array(wealth_ls)
            wealth_sizes.sort()
            sizes = wealth_sizes[::-1][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.3)]
            # wealths = np.log(sizes)
            wealths = sizes
            (val, bins)  = np.histogram(wealths)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp_30pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            wealth_power_30pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            wealth_phase_30pc = math.log(mse_exp / mse_power)
        except:
            pass

        try:
            score_sizes = np.array(score_ls)
            score_sizes.sort()
            sizes = score_sizes[::-1][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.3)]
            # scores = np.log(sizes)
            scores = sizes
            (val, bins)  = np.histogram(scores)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            score_exp_30pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            score_power_30pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            score_phase_30pc = math.log(mse_exp / mse_power)

        except:
            pass


        try:
            wealth_dev_mean = np.std(wealth_ls) / np.mean(wealth_ls)
            score_dev_mean = np.std(score_ls) / np.mean(score_ls)
        except:
            wealth_dev_mean, score_dev_mean = 0, 0



    res = [wealth_dev_mean, score_dev_mean, wealth_exp_10pc, wealth_power_10pc, wealth_phase_10pc, score_exp_10pc, score_power_10pc, score_phase_10pc, wealth_exp_30pc, wealth_power_30pc, wealth_phase_30pc, score_exp_30pc, score_power_30pc, score_phase_30pc]

    return res

# res

num_families = 300
iteration = 30000
iteration = 10000
trial = 1
l = 300
trial = 2
r = 0.3
eta = 1 / num_families


r = [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.002], [0.2, 0.1]][int(sys.argv[1]) // 5][int(sys.argv[2])]
# r = [0.3, 0.5][int(sys.argv[2])]
for l in [[3, 5, 80, 10000], [10, 20, 70, 5000], [30, 50, 1000, 4000], [100, 200, 8, 2000], [300, 500, 7, 3000]][int(sys.argv[1]) % 5]:
    df_res = pd.DataFrame(index = ["exchange", "interest", "num_families", "wealth_gini", "score_gini", "wealth_exp_5pc", "wealth_power_5pc", "wealth_phase_5pc", "score_exp_5pc", "score_power_5pc", "score_phase_5pc", "wealth_exp_10pc", "wealth_power_10pc", "wealth_phase_10pc", "score_exp_10pc", "score_power_10pc", "score_phase_10pc", "wealth_exp_30pc", "wealth_power_30pc", "wealth_phase_30pc", "score_exp_30pc", "score_power_30pc", "score_phase_30pc"])
    path = f"{num_families}fam_interest{round(r * 1000)}pm_{l}exchange"
    for trial in range(5):
        try:
            res = main()
            params = [l, r, num_families]
            params.extend(res)
            df_res[len(df_res.columns)] = params
        except:
            pass
    df_res.to_csv(f"res/res_{path}.csv")
