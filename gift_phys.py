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

class State:
    def __init__(self, families):
        self.families = families
        self.network = np.ones([num_families, num_families])
        # self.df = pd.DataFrame()

class Family:
    def __init__(self, family_id,  wealth, status, debt, subordinates):
        self.family_id = family_id
        self.wealth = wealth
        self.status = status
        self.debt = debt
        self.subordinates = subordinates
        self.give = 0
        self.score = 0
        self.lifetime = 0

def generation(families, network):
    family_ids = list(range(num_families))
    random.shuffle(family_ids)
    donation_ls = []
    # recipient_id_ls = random.choices(family_ids, k = num_families)
    for doner_id in family_ids:
        doner = families[doner_id]
        if doner.wealth  > 0:
            recipient_id = random.choices(family_ids, weights = network[:, doner_id], k= 1)[0]
            if doner_id != recipient_id:
                recipient = families[recipient_id]
                # recipient.wealth +=  doner.wealth
                recipient.debt.append([doner, doner.wealth * r])
                doner.give = doner.wealth
                doner.wealth = 0
                recipient.score += 1
                network[recipient_id, doner_id] += 1
                donation_ls.append([recipient_id, doner_id])

    for family in families:
        # family.wealth += (1 + math.log(1 + family.wealth))
        family.wealth += 1
        # family.wealth += family.give - family.given
        family.wealth += family.give
        family.give = 0
        # family.given = 0

    unpayback_ls = donation_ls[:]
    random.shuffle(family_ids)
    for family_id in family_ids:
        family = families[family_id]
        if len(family.debt) == 0:
            family.status = 1
        else:
            count = 0
            remove_ls = []
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
                    owner.score += 1
                    network[owner.family_id, family_id] += 1
                    count += 1
                    if [family_id, owner.family_id] in unpayback_ls:
                        unpayback_ls.remove([family_id, owner.family_id])
                else:
                    owner.wealth += family.wealth
                    debt[1] -= family.wealth
                    family.wealth = 0
                    owner.score += 1
                    network[owner.family_id, family_id] += 1
            family.debt = family.debt[count:]
            if len(family.debt) == 0:
                family.status = 1
            else:
                family.status = 0
                for debt in family.debt:
                    owner = debt[0]
                    if not family in owner.subordinates:
                        owner.subordinates.append(family)

    if len(donation_ls) > 0:
        payback = 1  - len(unpayback_ls) / len(donation_ls)
    else:
        payback = 0

    statuses = [family.status for family in families]
    # wealths = [family.wealth for family in families]
    # scores = [family.score for family in families]
    wealths = []
    scores = []
    given_ls = []

    for family in families:
        family.lifetime += 1
        if random.random() < 1 / l:
            wealths.append(family.wealth)
            scores.append(family.score)
            family.wealth = 0.0
            family.states = 1
            family.score = 0
            family.debt = []
            family.subordinates = []
            family.lifetime = 0
            network[family.family_id, :] = 1
            network[:, family.family_id] = 1

    return families, network, statuses, wealths, scores, payback


def mean(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0

np.sum(network, axis = 1) - num_families == scores


def main():
    families = [Family(i,  0.0,  1, [], []) for i in range(num_families)]
    network = np.ones([num_families, num_families])

    # independent_duration_res, subordinate_duration_res, rich_duration_res = [], [], []
    wealth_ls, score_ls = [], []
    iter = 0
    while iter < iteration:
        # print(iter)
        families, network, statuses, wealths, scores, payback = generation(families, network)
        iter += 1
        if iter >= iteration * 0.9:
            wealth_ls.extend(wealths)
            score_ls.extend(scores)

    if iter == iteration and trial < 3:
        try:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(wealth_ls, bins = 50, density = 1)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log.pdf")
            plt.close('all')

            wealth_ls = np.array(wealth_ls)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(wealth_ls[wealth_ls < 50], bins = 50, density = 1)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_exp.pdf")
            plt.close('all')

            max_val = max(10, np.max(wealth_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(wealth_ls, bins = np.logspace(0, np.log10(np.max(wealth_ls)), 50), density = 1)
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

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls, bins = 50, density = 1)
            ax.set_xlabel("score",fontsize=24)
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log.pdf")
            plt.close('all')

            score_ls = np.array(score_ls)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls[score_ls < 50], bins = 50, density = 1)
            ax.set_xlabel("score",fontsize=24)
            ax.set_yscale('log')
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_exp.pdf")
            plt.close('all')

            max_val = max(1, np.max(score_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls, bins = np.logspace(0, np.log10(np.max(score_ls)), 50), density = 1)
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

        wealth_exp_5pc, wealth_power_5pc, wealth_phase_5pc = np.nan, np.nan, np.nan
        score_exp_5pc, score_power_5pc, score_phase_5pc = np.nan, np.nan, np.nan
        try:
            wealth_sizes = np.array(wealth_ls)
            wealth_sizes.sort()
            sizes = wealth_sizes[::-1][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.05)]
            # wealths = np.log(sizes)
            wealths = sizes
            (val, bins)  = np.histogram(wealths)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp_5pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            wealth_power_5pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            wealth_phase_5pc = math.log(mse_exp / mse_power)
        except:
            pass

        try:
            score_sizes = np.array(score_ls)
            score_sizes.sort()
            sizes = score_sizes[::-1][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.05)]
            # scores = np.log(sizes)
            scores = sizes
            (val, bins)  = np.histogram(scores)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            score_exp_5pc = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            score_power_5pc = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            score_phase_5pc = math.log(mse_exp / mse_power)

        except:
            pass

        wealth_gini = gini2(wealth_ls)
        score_gini = gini2(score_ls)

    res = [wealth_gini, score_gini, wealth_exp_5pc, wealth_power_5pc, wealth_phase_5pc, score_exp_5pc, score_power_5pc, score_phase_5pc, wealth_exp_10pc, wealth_power_10pc, wealth_phase_10pc, score_exp_10pc, score_power_10pc, score_phase_10pc, wealth_exp_30pc, wealth_power_30pc, wealth_phase_30pc, score_exp_30pc, score_power_30pc, score_phase_30pc]

    return res

num_families = 3000
iteration = 10000
iteration = 100000
trial = 0
l = 100
trial = 1
r = 0.1


# r = [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.002], [0.2, 0.1]][int(sys.argv[1]) // 5][int(sys.argv[2])]
# # r = [0.3, 0.5][int(sys.argv[2])]
# for l in [[3, 5, 80], [10, 20, 70], [30, 50, 1000], [100, 200, 8], [300, 500, 7]][int(sys.argv[1]) % 5]:
r = [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.002], [0.2, 0.1]][int(sys.argv[1]) // 5][int(sys.argv[2])]
# r = [0.3, 0.5][int(sys.argv[2])]
for l in [[2000], [3000], [5000], [10000], [4000]][int(sys.argv[1]) % 5]:
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
