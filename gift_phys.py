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

import networkx as nx

if int(sys.argv[1])==0:
    if not os.path.exists("./res"):
        os.mkdir("./res")
        os.mkdir("./figs")

def linear_fit(x, a, b):
    return a * x + b

class Family:
    def __init__(self, family_id,  wealth, debt, subordinates):
        self.family_id = family_id
        self.wealth = wealth
        self.debt = debt
        self.subordinates = subordinates
        self.give = 0
        # self.given = 0
        self.score = 0

def generation(families, network):
    family_ids_ = list(range(num_families))
    family_ids = list(range(num_families))
    random.shuffle(family_ids)
    donation_ls = []
    # recipient_id_ls = random.choices(family_ids, k = num_families)
    for doner_id in family_ids:
        doner = families[doner_id]
        if len(doner.debt) == 0:
            recipient_id = random.choices(family_ids_, weights = network[:, doner_id], k= 1)[0]
            # recipient_id = recipient_id_ls[doner_id]
            if doner_id != recipient_id:
                recipient = families[recipient_id]
                if recipient not in doner.subordinates:
                    # recipient.wealth +=  doner.wealth
                    recipient.debt.append([doner, doner.wealth * r])
                    doner.give = doner.wealth
                    # recipient.given = doner.wealth
                    doner.wealth = 0
                    recipient.score += 1 / l
                    network[recipient_id, doner_id] += 1
                    donation_ls.append([recipient_id, doner_id])

    # given_ls = [family.given for family in families]
    for family in families:
        # family.wealth += (1 + math.log(1 + family.wealth))
        family.wealth += 1 / l
        # family.wealth += family.give - family.given
        family.wealth += family.give
        family.give = 0
        # family.given = 0

    random.shuffle(family_ids)
    for family_id in family_ids:
        family = families[family_id]
        if len(family.debt) > 0:
            count = 0
            remove_ls = []
            random.shuffle(family.debt)
            for debt in family.debt:
                owner = debt[0]
                debt_wealth = debt[1]
                if family not in owner.subordinates and [family.family_id, owner.family_id] not in donation_ls:
                    remove_ls.append(debt)
                    continue
                elif family.wealth >= debt_wealth:
                    owner.wealth += debt_wealth
                    if family in owner.subordinates:
                        owner.subordinates.remove(family)
                    family.wealth -= debt_wealth
                    # owner.score += 1 / l / len(family.debt)
                    owner.score += 1 / l
                    network[owner.family_id, family_id] += 1
                    remove_ls.append(debt)
                else:
                    owner.wealth += family.wealth
                    debt[1] -= family.wealth
                    family.wealth = 0
                    # owner.score += 1 / l / len(family.debt)
                    owner.score += 1 / l
                    network[owner.family_id, family_id] += 1
            for removed in remove_ls:
                family.debt.remove(removed)
            for debt in family.debt:
                owner = debt[0]
                if not family in owner.subordinates:
                    owner.subordinates.append(family)

    wealths = [family.wealth for family in families]
    scores = [family.score for family in families]
    wealths = sorted(wealths)
    wealth_ratio = wealths[-1] / wealths[-2]
    top_wealth = wealths[-1]
    # top_wealth = len([0 for family in families if len(family.debt) > 0]) / num_families
    # wealths = []
    # scores = []

    for family in families:
        if random.random() < 1 / l:
            # wealths.append(family.wealth)
            # scores.append(family.score)
            family.wealth = 1 / l
            family.score = 0
            family.debt = []
            family.subordinates = []
            network[family.family_id, :] = 1
            network[:, family.family_id] = 1

    return families, network,  wealths, scores, wealth_ratio, top_wealth


def mean(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0

def main():
    families = [Family(i,  1 / l,  [], []) for i in range(num_families)]
    network = np.ones([num_families, num_families])

    # independent_duration_res, subordinate_duration_res, rich_duration_res = [], [], []
    wealth_ls, score_ls, wealth_ratio_ls, top_wealth_ls = [], [], [], []
    tot_wealth, tot_score = [], []
    # given_ls = []
    iter = 0
    while iter < iteration:
        # print(iter)
        families, network,  wealths, scores, wealth_ratio, top_wealth = generation(families, network)
        tot_wealth.append(sum(wealths))
        tot_score.append(sum(scores))
        iter += 1
        if iter >= iteration * 0.9:
            wealth_ls.extend(wealths)
            score_ls.extend(scores)
            wealth_ratio_ls.append(wealth_ratio)
            top_wealth_ls.append(top_wealth)
            # given_ls.extend(given)

    wealths = [family.wealth for family in families]
    nearest_wealths = []
    for i in range(num_families):
        j = np.argmax(network[:, i])
        nearest_wealths.append([wealths[i], wealths[j]])

    nearest_wealths = np.array(nearest_wealths)
    nearest_corr = np.corrcoef(nearest_wealths.T)[0, 1]

    if iter == iteration and trial < 2:
        try:
            # statuses_c = []
            # for family in families:
            #     if len(family.debt) == 0:
            #         statuses_c.append("b")
            #     else:
            #         statuses_c.append("m")
            #
            # cur_connection = 1 * (network > network.mean(axis = 0) + network.var(axis = 0) ** (1/2))
            # # cur_connection = 1 * (connection > 0.08)
            # df = pd.DataFrame(cur_connection)
            # G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())
            # remove_ls = []
            # count = 0
            # for v in G:
            #     G_deg=G.degree(v)
            #     if G_deg==0:
            #         remove_ls.append(v)
            #         statuses_c.remove(statuses_c[count])
            #     count += 1
            # for v in remove_ls:
            #     G.remove_node(v)
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # nx.draw_networkx(G, node_size = 30, with_labels=False, ax = ax, node_color = statuses_c)
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_network.pdf")
            # plt.close('all')
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # # ax.plot(np.arange(int(iteration * 0.9) - 1, iteration, 1), tot_wealth)
            # ax.scatter(nearest_wealths[:, 0], nearest_wealths[:, 1])
            # ax.set_xlabel("donor",fontsize=24)
            # ax.set_ylabel("recipient",fontsize=24)
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_nearest.pdf")
            # plt.close('all')
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # # ax.plot(np.arange(int(iteration * 0.9) - 1, iteration, 1), tot_wealth)
            # ax.plot(tot_wealth)
            # ax.set_xlabel("total wealth",fontsize=24)
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_tot_wealth.pdf")
            # plt.close('all')
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ax.plot(tot_score)
            # ax.set_xlabel("total score",fontsize=24)
            # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # # ax.set_ylim(-0.1,1.5)
            # ax.tick_params(labelsize=18)
            # fig.tight_layout()
            # fig.savefig(f"figs/{path}_{trial}_tot_score.pdf")
            # plt.close('all')

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

            x = np.linspace(0.01, np.max(wealth_ls), 50)
            rl = r * l
            y = (1 - rl)**(1 /rl) / (1 + (x-1) * rl)**(1 + 1 / rl)
            y = y / np.sum(y)
            b = (1 + math.sqrt(1 + 4 * rl)) / 2
            z = np.exp(- x / b)
            z = z / np.sum(z)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            # ax.hist(wealth_ls, bins = 50, density = 1)
            ax.hist(wealth_ls, bins = 50, weights = weights)
            ax = sns.lineplot(x, y, color = "black")
            ax = sns.lineplot(x, z, color = "red")
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log2.pdf")
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

            max_val = max(10, np.max(wealth_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(wealth_ls, bins = np.logspace(0, np.log10(np.max(wealth_ls)), 50),weights = weights)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, max_val + 10)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=22)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_log2.pdf")
            plt.close('all')

            sx = sorted(wealth_ls)[::-1]
            sx = np.array(sx)
            N = len(wealth_ls)
            sy = [i/N for i in range(N)]

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx[int(len(sx) * 0.001):], sy[int(len(sy) * 0.001):])
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_wealth_log_CDF.pdf")
            plt.close('all')

            x_min = max(max(sx[sx > 0][-1] * 0.9, sx[int(len(sx) / 2)]), 1e-02)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("wealth",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(x_min, sx[0] + 1)
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

            max_val = max(10, np.max(score_ls))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(score_ls, bins = np.logspace(0, np.log10(np.max(score_ls)), 50),weights = weights)
            ax.set_xlabel("score",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, max_val + 10)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=22)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_log2.pdf")
            plt.close('all')

            sx = sorted(score_ls)[::-1]
            sx = np.array(sx)
            N = len(score_ls)
            sy = [i/N for i in range(N)]


            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx[int(len(sx) * 0.001):], sy[int(len(sy) * 0.001):])
            ax.set_xlabel("score",fontsize=24)
            ax.set_yscale('log')
            plt.xticks(rotation=45)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=18)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_score_log_CDF.pdf")
            plt.close('all')

            x_min = max(max(sx[sx > 0][-1] * 0.9, sx[int(len(sx) / 2)]), 1e-02)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sx, sy)
            ax.set_xlabel("score",fontsize=24)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(x_min, sx[0] + 1)
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
            wealth_sizes = sorted(wealth_ls)[::-1]
            N = len(wealth_ls)
            sy = [(i + 1) / N for i in range(N)][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.5)]
            sizes = np.array(wealth_sizes[int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.5)])
            param, cov = curve_fit(linear_fit, sizes, np.log(sy))
            wealth_exp_10pc = - 1 / param[0]
            predict = np.exp(sizes * param[0])
            mse_exp = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)

            sizes = sizes - 1 + 1 / r / l
            sizes_ = np.where(sizes <= 0, sizes[sizes > 0].min(), sizes)
            param, cov = curve_fit(linear_fit, np.log(sizes_), np.log(sy))
            wealth_power_10pc = - param[0] + 1
            predict = sizes_ ** (param[0])
            mse_power = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)
            wealth_phase_10pc = math.log(mse_exp / mse_power)
        except:
            pass

        try:
            score_sizes = sorted(score_ls)[::-1]
            N = len(score_ls)
            sy = [(i + 1) / N for i in range(N)][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.5)]
            sizes = np.array(score_sizes[int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.5)])
            param, cov = curve_fit(linear_fit, sizes, np.log(sy))
            score_exp_10pc = - 1 / param[0]
            predict = np.exp(sizes * param[0])
            mse_exp = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(sizes), np.log(sy))
            score_power_10pc = - param[0] + 1
            predict = sizes ** (param[0])
            mse_power = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)
            score_phase_10pc = math.log(mse_exp / mse_power)
        except:
            pass

        wealth_exp_30pc, wealth_power_30pc, wealth_phase_30pc = np.nan, np.nan, np.nan
        score_exp_30pc, score_power_30pc, score_phase_30pc = np.nan, np.nan, np.nan
        try:
            wealth_sizes = sorted(wealth_ls)[::-1]
            N = len(wealth_ls)
            sy = [(i + 1) / N for i in range(N)][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.3)]
            sizes = np.array(wealth_sizes[int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.3)])
            param, cov = curve_fit(linear_fit, sizes, np.log(sy))
            wealth_exp_30pc = - 1 / param[0]
            predict = np.exp(sizes * param[0])
            mse_exp = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)

            sizes = sizes - 1 + 1 / r / l
            sizes_ = np.where(sizes <= 0, sizes[sizes > 0].min(), sizes)
            param, cov = curve_fit(linear_fit, np.log(sizes_), np.log(sy))
            wealth_power_30pc = - param[0] + 1
            predict = sizes_ ** (param[0])
            mse_power = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)
            wealth_phase_30pc = math.log(mse_exp / mse_power)
        except:
            pass

        try:
            score_sizes = sorted(score_ls)[::-1]
            N = len(score_ls)
            sy = [(i + 1) / N for i in range(N)][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.3)]
            sizes = np.array(score_sizes[int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.3)])
            param, cov = curve_fit(linear_fit, sizes, np.log(sy))
            score_exp_30pc = - 1 / param[0]
            predict = np.exp(sizes * param[0])
            mse_exp = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(sizes), np.log(sy))
            score_power_30pc = - param[0] + 1
            predict = sizes ** (param[0])
            mse_power = np.mean((sy / np.sum(sy) - predict / np.sum(predict)) ** 2)
            score_phase_30pc = math.log(mse_exp / mse_power)

        except:
            pass

        wealth_exp_10pc_pdf, wealth_power_10pc_pdf, wealth_phase_10pc_pdf = np.nan, np.nan, np.nan
        score_exp_10pc_pdf, score_power_10pc_pdf, score_phase_10pc_pdf = np.nan, np.nan, np.nan

        try:
            wealth_sizes = np.array(wealth_ls)
            wealth_sizes.sort()
            sizes = wealth_sizes[::-1][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.5)]
            # wealths = np.log(sizes)
            wealths = sizes
            (val, bins)  = np.histogram(wealths)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp_10pc_pdf = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            wealth_power_10pc_pdf = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            wealth_phase_10pc_pdf = math.log(mse_exp / mse_power)
        except:
            pass

        try:
            score_sizes = np.array(score_ls)
            score_sizes.sort()
            sizes = score_sizes[::-1][int(len(score_sizes) * 0.001): int(len(score_sizes) * 0.5)]
            # scores = np.log(sizes)
            scores = sizes
            (val, bins)  = np.histogram(scores)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            score_exp_10pc_pdf = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            score_power_10pc_pdf = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            score_phase_10pc_pdf = math.log(mse_exp / mse_power)

        except:
            pass

        wealth_exp_30pc_pdf, wealth_power_30pc_pdf, wealth_phase_30pc_pdf = np.nan, np.nan, np.nan
        score_exp_30pc_pdf, score_power_30pc_pdf, score_phase_30pc_pdf = np.nan, np.nan, np.nan
        try:
            wealth_sizes = np.array(wealth_ls)
            wealth_sizes.sort()
            sizes = wealth_sizes[::-1][int(len(wealth_sizes) * 0.001): int(len(wealth_sizes) * 0.3)]
            # wealths = np.log(sizes)
            wealths = sizes
            (val, bins)  = np.histogram(wealths)
            bins = (bins[1:] + bins[:-1]) / 2
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp_30pc_pdf = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            wealth_power_30pc_pdf = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            wealth_phase_30pc_pdf = math.log(mse_exp / mse_power)
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
            score_exp_30pc_pdf = - 1 / param[0]
            predict = np.exp(bins * param[0])
            mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

            param, cov = curve_fit(linear_fit, np.log(bins), np.log(val))
            score_power_30pc_pdf = - param[0]
            predict = bins ** (param[0])
            mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)
            score_phase_30pc_pdf = math.log(mse_exp / mse_power)

        except:
            pass


        try:
            wealth_dev_mean = np.std(wealth_ls) / np.mean(wealth_ls)
            score_dev_mean = np.std(score_ls) / np.mean(score_ls)
            pop_wealth = wealth_sizes[int(len(wealth_sizes) * 0.001):]
            wealth_dev_mean_pop = np.std(pop_wealth) / np.mean(pop_wealth)
            pop_score = score_sizes[int(len(score_sizes) * 0.001):]
            score_dev_mean_pop = np.std(pop_score) / np.mean(pop_score)

        except:
            wealth_dev_mean, score_dev_mean = 0, 0

            # wealth_ls = np.array(wealth_ls)
            # score_ls = np.array(score_ls)
            # wealth_ls_ = wealth_ls[wealth_ls > 0]
            # score_ls_ = score_ls[score_ls > 1]
            # wealth_dev_mean_ = np.std(wealth_ls_) / np.mean(wealth_ls_)
            # score_dev_mean_ = np.std(score_ls_) / np.mean(score_ls_)
            # print(wealth_dev_mean, score_dev_mean, wealth_dev_mean_, score_dev_mean_)
            # len(score_ls[score_ls < 1])

    wealth_ratio = np.mean(np.array(wealth_ratio_ls))
    top_wealth = np.mean(np.array(top_wealth_ls))

    res = [nearest_corr, wealth_dev_mean, score_dev_mean, wealth_dev_mean_pop, score_dev_mean_pop, wealth_ratio, top_wealth, wealth_exp_10pc, wealth_power_10pc, wealth_phase_10pc, score_exp_10pc, score_power_10pc, score_phase_10pc, wealth_exp_30pc, wealth_power_30pc, wealth_phase_30pc, score_exp_30pc, score_power_30pc, score_phase_30pc, wealth_exp_10pc_pdf, wealth_power_10pc_pdf, wealth_phase_10pc_pdf, score_exp_10pc_pdf, score_power_10pc_pdf, score_phase_10pc_pdf, wealth_exp_30pc_pdf, wealth_power_30pc_pdf, wealth_phase_30pc_pdf, score_exp_30pc_pdf, score_power_30pc_pdf, score_phase_30pc_pdf]

    return res

num_families = 100
iteration = 10000
iteration = 100000
trial = 0
l = 20
trial = 0
r = 0.04

# for r in [0.003, 0.03, 0.3, 0.5]:
#     path = f"{num_families}fam_interest{round(r * 1000)}pm_{l}exchange"
#     print(main())
# df_res
# (1 + math.sqrt(1.4)) / 2
#
# for [r, l] in [[0.001, 100], [0.005, 20],[0.01, 10], [0.02, 5]]:
#     path = f"{num_families}fam_interest{round(r * 1000)}pm_{l}exchange"
#     res = main()
#     params = [l, r, num_families]
#     params.extend(res)
#     df_res[len(df_res.columns)] = params
# df_res
# 0.8 / 1.7
# for i in range(27):
#     rl = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0][i // 3]
#     for l in [[50, 1000], [100, 500], [200, 300]][i % 3]:
#         r = rl / l
#         print(r, l)

# rl = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000][int(sys.argv[1]) // 3]
r = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.3, 0.5, 0.2, 0.1, 0.001, 0.002][int(sys.argv[1]) % 12]
l = [10, 50000, 20, 30000, 30, 20000, 50, 10000, 100, 5000, 200, 3000, 300, 2000, 500, 1000][int(sys.argv[1]) // 12]
# for r in [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.5], [0.2, 0.1], [0.001, 0.002]][int(sys.argv[1]) % 6]:
#     # r = [0.3, 0.5][int(sys.argv[2])]
#     for l in [[10, 50000], [20, 30000], [30, 20000], [50, 10000], [100, 5000], [200, 3000], [300, 2000], [500, 1000]][int(sys.argv[1]) // 6]:
df_res = pd.DataFrame(index = ["exchange", "interest", "num_families", "nearest corr.", "wealth_gini", "score_gini", "wealth_gini2", "score_gini2", "wealth_ratio", "top_wealth", "wealth_exp_10pc", "wealth_power_10pc", "wealth_phase_10pc", "score_exp_10pc", "score_power_10pc", "score_phase_10pc", "wealth_exp_30pc", "wealth_power_30pc", "wealth_phase_30pc", "score_exp_30pc", "score_power_30pc", "score_phase_30pc", "wealth_exp_10pc_pdf", "wealth_power_10pc_pdf", "wealth_phase_10pc_pdf", "score_exp_10pc_pdf", "score_power_10pc_pdf", "score_phase_10pc_pdf", "wealth_exp_30pc_pdf", "wealth_power_30pc_pdf", "wealth_phase_30pc_pdf", "score_exp_30pc_pdf", "score_power_30pc_pdf", "score_phase_30pc_pdf"])
path = f"{num_families}fam_r{round(r * 10000)}pm_{l}exchange"
if not os.path.exists(f"res/res_{path}.csv"):
    for trial in range(100):
        try:
            res = main()
            params = [l, r, num_families]
            params.extend(res)
            df_res[len(df_res.columns)] = params
        except:
            pass
    df_res.to_csv(f"res/res_{path}.csv")

# for r in [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.5], [0.2, 0.1], [0.001, 0.002]][int(sys.argv[1]) % 6]:
#     # r = [0.3, 0.5][int(sys.argv[2])]
#     for l in [[10, 50000], [20, 30000], [30, 20000], [50, 10000], [100, 5000], [200, 3000], [300, 2000], [500, 1000]][int(sys.argv[1]) // 6]:
#         df_res = pd.DataFrame(index = ["exchange", "interest", "num_families", "nearest corr.", "wealth_gini", "score_gini", "wealth_gini2", "score_gini2", "wealth_ratio", "top_wealth", "wealth_exp_10pc", "wealth_power_10pc", "wealth_phase_10pc", "score_exp_10pc", "score_power_10pc", "score_phase_10pc", "wealth_exp_30pc", "wealth_power_30pc", "wealth_phase_30pc", "score_exp_30pc", "score_power_30pc", "score_phase_30pc", "wealth_exp_10pc_pdf", "wealth_power_10pc_pdf", "wealth_phase_10pc_pdf", "score_exp_10pc_pdf", "score_power_10pc_pdf", "score_phase_10pc_pdf", "wealth_exp_30pc_pdf", "wealth_power_30pc_pdf", "wealth_phase_30pc_pdf", "score_exp_30pc_pdf", "score_power_30pc_pdf", "score_phase_30pc_pdf"])
#         path = f"{num_families}fam_r{round(r * 10000)}pm_{l}exchange"
#         if not os.path.exists(f"res/res_{path}.csv"):
#             for trial in range(100):
#                 try:
#                     res = main()
#                     params = [l, r, num_families]
#                     params.extend(res)
#                     df_res[len(df_res.columns)] = params
#                 except:
#                     pass
#             df_res.to_csv(f"res/res_{path}.csv")

# r = [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.5], [0.2, 0.1]][int(sys.argv[1]) // 5][int(sys.argv[2])]
# # for r in [[0.003, 0.005], [0.01, 0.02], [0.03, 0.05], [0.3, 0.5], [0.2, 0.1]][int(sys.argv[1]) // 5]:
#     # r = [0.3, 0.5][int(sys.argv[2])]
# for l in [[3, 5, 80, 10000], [10, 20, 70, 5000], [30, 50, 1000, 4000], [100, 200, 8, 2000], [300, 500, 7, 3000]][int(sys.argv[1]) % 5]:
#     df_res = pd.DataFrame(index = ["exchange", "interest", "num_families", "nearest corr.", "wealth_gini", "score_gini", "wealth_gini2", "score_gini2", "wealth_ratio", "top_wealth", "wealth_exp_10pc", "wealth_power_10pc", "wealth_phase_10pc", "score_exp_10pc", "score_power_10pc", "score_phase_10pc", "wealth_exp_30pc", "wealth_power_30pc", "wealth_phase_30pc", "score_exp_30pc", "score_power_30pc", "score_phase_30pc", "wealth_exp_10pc_pdf", "wealth_power_10pc_pdf", "wealth_phase_10pc_pdf", "score_exp_10pc_pdf", "score_power_10pc_pdf", "score_phase_10pc_pdf", "wealth_exp_30pc_pdf", "wealth_power_30pc_pdf", "wealth_phase_30pc_pdf", "score_exp_30pc_pdf", "score_power_30pc_pdf", "score_phase_30pc_pdf"])
#     path = f"{num_families}fam_rl{round(rl * 1000)}pm_{l}exchange"
#     for trial in range(30):
#         try:
#             res = main()
#             params = [l, r, num_families]
#             params.extend(res)
#             df_res[len(df_res.columns)] = params
#         except:
#             pass
#     df_res.to_csv(f"res/res_{path}.csv")
