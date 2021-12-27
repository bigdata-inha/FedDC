# from __future__ import print_function
# import argparse
# import pickle
# import numpy as np
# from sklearn.utils.extmath import randomized_svd
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial.distance import cosine
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.spatial import distance
# import sys
# import os
# import scipy
# import random
# from torch.utils.tensorboard import SummaryWriter
# from collections import defaultdict
# from sklearn import linear_model
# import statsmodels.api as sm
# import cvxpy as cp
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from scipy.optimize import minimize
# from scipy.optimize import least_squares
#
# from scipy.optimize import lsq_linear
# import Ridge
#
# cwd = os.getcwd()
# sys.path.append(cwd + '/../')
#
#
# def create_scaling_mat_ip_thres_bias(weight, ind, threshold, model_type):
#     '''
#     weight - 2D matrix (n_{i+1}, n_i), np.ndarray
#     ind - chosen indices to remain, np.ndarray
#     threshold - cosine similarity threshold
#     '''
#     assert (type(weight) == np.ndarray)
#     assert (type(ind) == np.ndarray)
#
#     cosine_sim = 1 - pairwise_distances(weight, metric="cosine")
#     weight_chosen = weight[ind, :]
#     scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])
#
#     for i in range(weight.shape[0]):
#         if i in ind:  # chosen
#             ind_i, = np.where(ind == i)
#             assert (len(ind_i) == 1)  # check if only one index is found
#             scaling_mat[i, ind_i] = 1
#         else:  # not chosen
#             if model_type == 'prune':
#                 continue
#             max_cos_value = np.max(cosine_sim[i][ind])
#             max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]
#
#             if threshold and max_cos_value < threshold:
#                 continue
#
#             baseline_weight = weight_chosen[max_cos_value_index]
#             current_weight = weight[i]
#             baseline_norm = np.linalg.norm(baseline_weight)
#             current_norm = np.linalg.norm(current_weight)
#             scaling_factor = current_norm / baseline_norm
#             scaling_mat[i, max_cos_value_index] = scaling_factor
#             scaling_mat[i, max_cos_value_index] = scaling_factor
#
#     return scaling_mat
#
#
# def create_scaling_mat_conv_thres_bn(weight, ind, threshold,
#                                      bn_weight, bn_bias,
#                                      bn_mean, bn_var, lam, model_type):
#     '''
#     weight - 4D tensor(n, c, h, w), np.ndarray
#     ind - chosen indices to remain
#     threshold - cosine similarity threshold
#     bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
#     bn_mean, bn_var - running_mean, running_var of BN (for inference)
#     lam - how much to consider cosine sim over bias, float value between 0 and 1
#     '''
#     assert (type(weight) == np.ndarray)
#     assert (type(ind) == np.ndarray)
#     assert (type(bn_weight) == np.ndarray)
#     assert (type(bn_bias) == np.ndarray)
#     assert (type(bn_mean) == np.ndarray)
#     assert (type(bn_var) == np.ndarray)
#     assert (bn_weight.shape[0] == weight.shape[0])
#     assert (bn_bias.shape[0] == weight.shape[0])
#     assert (bn_mean.shape[0] == weight.shape[0])
#     assert (bn_var.shape[0] == weight.shape[0])
#
#     weight = weight.reshape(weight.shape[0], -1)
#
#     cosine_dist = pairwise_distances(weight, metric="cosine")
#
#     weight_chosen = weight[ind, :]
#     scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])  # 16,11
#
#     for i in range(weight.shape[0]):  # 16
#         if i in ind:  # chosen
#             ind_i, = np.where(ind == i)
#             assert (len(ind_i) == 1)  # check if only one index is found
#             scaling_mat[i, ind_i] = 1
#         else:  # not chosen
#
#             if model_type == 'prune':
#                 continue
#
#             current_weight = weight[i]
#             current_norm = np.linalg.norm(current_weight)
#             current_cos = cosine_dist[i]
#             gamma_1 = bn_weight[i]
#             beta_1 = bn_bias[i]
#             mu_1 = bn_mean[i]
#             sigma_1 = bn_var[i]
#
#             # choose one
#             cos_list = []
#             scale_list = []
#             bias_list = []
#
#             for chosen_i in ind:
#                 chosen_weight = weight[chosen_i]
#                 chosen_norm = np.linalg.norm(chosen_weight, ord=2)
#                 chosen_cos = current_cos[chosen_i]
#                 gamma_2 = bn_weight[chosen_i]
#                 beta_2 = bn_bias[chosen_i]
#                 mu_2 = bn_mean[chosen_i]
#                 sigma_2 = bn_var[chosen_i]
#
#                 # compute cosine sim
#                 cos_list.append(chosen_cos)
#
#                 # compute s
#                 s = current_norm / chosen_norm
#
#                 # compute scale term
#                 scale_term_inference = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
#                 scale_list.append(scale_term_inference)
#
#                 # compute bias term
#                 bias_term_inference = abs(
#                     (gamma_2 / sigma_2) * (s * (-(sigma_1 * beta_1 / gamma_1) + mu_1) - mu_2) + beta_2)
#
#                 bias_term_inference = bias_term_inference / scale_term_inference
#
#                 bias_list.append(bias_term_inference)
#
#             assert (len(cos_list) == len(ind))
#             assert (len(scale_list) == len(ind))
#             assert (len(bias_list) == len(ind))
#
#             # merge cosine distance and bias distance
#             bias_list = (bias_list - np.min(bias_list)) / (np.max(bias_list) -
#                                                            np.min(bias_list))
#
#             score_list = lam * np.array(cos_list) + (1 - lam) * np.array(bias_list)
#             # score_list =  np.array(cos_list)
#
#             # find index and scale with minimum distance
#             min_ind = np.argmin(score_list)
#
#             min_scale = scale_list[min_ind]
#             min_cosine_sim = 1 - cos_list[min_ind]
#
#             # check threshold - second
#             if threshold and min_cosine_sim < threshold:
#                 continue
#
#             scaling_mat[i, min_ind] = min_scale
#
#             # for index, scale in enumerate(scale_list):
#             #     scaling_mat[i, index] = scale
#
#     return scaling_mat
#
#
# def ours_create_scaling(weight, ind, bn_weight, bn_bias,
#                         bn_mean, bn_var, lam, lam_2, compensation_threshold):
#     def loss_fn(X, Y, alpha_list):
#
#         return cp.norm2(X @ alpha_list - Y) ** 2
#         # return cp.norm1(X @ alpha_list - Y) #** 2
#
#     def regularizer(alpha_list, alpha_scale_list, gamma_1, sigma_1, mu_1, beta_1):
#
#         regularization_term = gamma_1 / sigma_1 * cp.sum(alpha_list * alpha_scale_list) - (
#                     gamma_1 / sigma_1 * mu_1) + beta_1
#
#         return cp.pnorm(regularization_term, p=2) ** 2
#         # return cp.norm1(regularization_term)
#
#     def Ridge_regularizer(alpha_list):
#         # return cp.norm1(alpha_list)
#         return cp.pnorm(alpha_list, p=2) ** 2
#
#     def R_squared(X, Y, alpha_list):
#         SSE = loss_fn(X, Y, alpha_list).value
#         SST = np.sum((Y - np.mean(Y)) ** 2)
#         return 1 - SSE / SST
#
#     def objective_fn(X, Y, alpha_list, alpha_scale_list, gamma_1, sigma_1, mu_1, beta_1, lam, lam_2):
#         # (l1norm- (0.01,0.005) - 87.65 | (0.001,0.01) - 87.74, (0.001,0.008) - 87.76 , (0.001,0.009) - 87.67)
#         # (l2norm - (0.01, 0.01) :87.5, (0.01,0.005) - 87.52 ,(0.001,0.01) : 87.76 , (0.001,0.008) - 87.74, (0.001,0.009) - 87.79)
#         # print( (sigma_1/gamma_1)**2) : small value
#
#         return loss_fn(X, Y, alpha_list) + lam * (sigma_1 / gamma_1) ** 2 * regularizer(alpha_list, alpha_scale_list,
#                                                                                         gamma_1, sigma_1, mu_1,
#                                                                                         beta_1) + lam_2 * Ridge_regularizer(
#             alpha_list)
#         # return loss_fn(X, Y, alpha_list) + lam * regularizer(alpha_list, alpha_scale_list, gamma_1, sigma_1, mu_1,beta_1) + lam_2 * lasso_regularizer(alpha_list)
#
#     def mse(X, Y, alpha_list):
#         return (1.0 / X.shape[0]) * loss_fn(X, Y, alpha_list).value
#
#     def SSE(X, Y, alpha_list):
#         return loss_fn(X, Y, alpha_list).value
#
#     weight = weight.reshape(weight.shape[0], -1)  # (16, -1)
#
#     weight_chosen = weight[ind, :]
#     scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])  # 16,11
#     error_dict = dict()
#     alpha_dict = dict()
#
#     for i in range(weight.shape[0]):  # 16
#         if i in ind:  # chosen
#             ind_i, = np.where(ind == i)
#             assert (len(ind_i) == 1)  # check if only one index is found
#             scaling_mat[i, ind_i] = 1
#         else:  # not chosen
#
#             # choose one
#             gamma_1 = bn_weight[i]
#             beta_1 = bn_bias[i]
#             mu_1 = bn_mean[i]
#             sigma_1 = bn_var[i]
#             preserved_weights = []
#
#             for chosen_i in ind:
#                 preserved_weights.append(weight[chosen_i])
#             preserved_weights = np.array(preserved_weights).T
#             # alpha_list = cp.Variable(preserved_weights.shape[1])
#             alpha_scale_list = []
#
#             # r_squared = linear_clf.score(np.array(preserved_weights), weight[i])
#
#             for index, chosen_i in enumerate(ind):
#                 gamma_2 = bn_weight[chosen_i]
#                 beta_2 = bn_bias[chosen_i]
#                 mu_2 = bn_mean[chosen_i]
#                 sigma_2 = bn_var[chosen_i]
#                 alpha_scale = (mu_2 - (sigma_2 * beta_2 / gamma_2))
#                 alpha_scale_list.append(alpha_scale)
#
#             # linear_clf = linear_model.Ridge(fit_intercept = False, alpha_scale_list=alpha_scale_list, sigma_1=sigma_1,gamma_1=gamma_1)
#             # linear_clf.fit(np.array(preserved_weights), weight[i])
#             # alpha_list = linear_clf.coef_
#
#             linear_cif = Ridge.Ridge_Regression(np.array(preserved_weights), weight[i], alpha=[lam, lam_2],
#                                                 fit_intercept=False, alpha_scale_list=alpha_scale_list, sigma_1=sigma_1,
#                                                 gamma_1=gamma_1, mu_1=mu_1, beta_1=beta_1)
#             alpha_list = linear_cif.fit()
#
#             # total_prob = cp.Problem(cp.Minimize(objective_fn(preserved_weights, weight[i], alpha_list, alpha_scale_list,
#             #                                                  gamma_1, sigma_1, mu_1, beta_1, lam,lam_2)))
#             # total_prob.solve()
#             sse = float(SSE(preserved_weights, weight[i], alpha_list))
#             error_dict[i] = sse
#             alpha_dict[i] = alpha_list
#
#             #             R_s = R_squared(preserved_weights, weight[i], alpha_list)
#             #             print(sse)
#
#             # print(R_square)
#
#             #             if sse < 0.24 : # sse 0.21 - 87.64
#             for index, chosen_i in enumerate(ind):
#                 gamma_2 = bn_weight[chosen_i]
#                 sigma_2 = bn_var[chosen_i]
#                 scaling_mat[i, index] = alpha_list[index] * (gamma_1 / gamma_2) * (sigma_2 / sigma_1)
#
#     return scaling_mat, error_dict, alpha_dict
#
#
# class Decompose:
#     def __init__(self, arch, param_dict, criterion, threshold, lamda, lamda_2, compensation_threshold, model_type, cfg,
#                  cuda):
#
#         self.param_dict = param_dict
#         self.arch = arch
#         self.criterion = criterion
#         self.threshold = threshold
#         self.lamda = lamda
#         self.lamda_2 = lamda_2
#         self.model_type = model_type
#         self.cfg = cfg
#         self.cuda = cuda
#         self.output_channel_index = {}
#         self.pruned_channel_index = {}
#         self.decompose_weight = []
#         self.conv1_norm_dictionary = dict()
#         self.conv2_norm_dictionary = dict()
#         self.error_dict = dict()
#         self.alpha_dict = dict()
#         self.compensation_threshold = compensation_threshold
#
#     def get_output_channel_index(self, value, layer_id):
#
#         output_channel_index = []
#
#         if len(value.size()):
#
#             weight_vec = value.view(value.size()[0], -1)
#             weight_vec = weight_vec.cuda()
#
#             # l1-norm
#             if self.criterion == 'l1-norm':
#                 norm = torch.norm(weight_vec, 1, 1)
#                 norm_np = norm.cpu().detach().numpy()
#                 arg_max = np.argsort(norm_np)
#                 arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
#                 output_channel_index = sorted(arg_max_rev.tolist())
#                 print('select_index', output_channel_index)
#
#             # l2-norm
#             elif self.criterion == 'l2-norm':
#
#                 norm = torch.norm(weight_vec, 2, 1)
#                 norm_np = norm.cpu().detach().numpy()
#                 arg_max = np.argsort(norm_np)
#                 arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
#                 output_channel_index = sorted(arg_max_rev.tolist())
#                 # print('select_index', value.size() , len(output_channel_index))
#
#             # l2-GM
#             elif self.criterion == 'l2-GM':
#                 weight_vec = weight_vec.cpu().detach().numpy()
#                 matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
#                 similar_sum = np.sum(np.abs(matrix), axis=0)
#                 output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]
#                 output_channel_index.sort()
# #                 print('select_index', output_channel_index)
#
#             elif self.criterion == 'random':
#                 select_index = random.sample(list(range(weight_vec.shape[0])), self.cfg[layer_id])
#                 print('select_index', select_index)
#                 output_channel_index = select_index
#
#             elif self.criterion == 'JSD':
#                 channel_js_dist = dict()
#                 select_index = random.sample(list(range(weight_vec.shape[0])), self.cfg[layer_id])
#
#                 for o_c in range(value.size(0)):
#                     k_channel_weight = list()
#
#                     for i_c in range(value.size(1)):
#                         X = value[o_c, i_c, :, :].reshape(-1).cpu().detach().numpy()
#                         # x, y1 = TreeKDE(bw="silverman").fit(X).evaluate(oriweight.size(1))
#                         # plt.hist(X)
#                         # plt.plot(x, y1, label='KDE estimate with defaults')
#                         # k_channel_weight.append(y1)
#                         k_channel_weight.append(X)
#                     js_sum = []
#                     for i in range(len(k_channel_weight)):
#                         for j in range(1 + i, len(k_channel_weight)):
#                             js_sum.append(distance.jensenshannon(k_channel_weight[i], k_channel_weight[j]))
#                             # js_sum.append(distance.euclidean(k_channel_weight[i], k_channel_weight[j]))
#                             # print(i,j)
#
#                     channel_js_dist[o_c] = np.sum(js_sum)
#                     # js_dist = np.sum([distance.jensenshannon(k_channel_weight[0],k_channel_weight[i]) for i in range(len(k_channel_weight))])
#
#                 c_sorted_dict = [key for key, _ in
#                                  sorted(channel_js_dist.items(), key=lambda x: x[1], reverse=False)]  # 오름차순
#
#                 count = len(channel_js_dist.keys()) - len(select_index)
#                 print('ORIGINAL_c_sorted_dict ', sorted(c_sorted_dict))
#                 c_sorted_dict.sort()
#                 del c_sorted_dict[:count]
#                 c_sorted_dict.sort()
#                 print('c_sorted_dict ', len(c_sorted_dict), c_sorted_dict)
#                 output_channel_index = c_sorted_dict
#
#         pruned_channel_index = list(set(list(range(weight_vec.shape[0]))) - set(output_channel_index))
#
#         return output_channel_index, np.array(pruned_channel_index)
#
#     #
#     # def OURS_get_channel(self, bn_weight, bn_bias, layer_id):
#     #     arg_max = np.argsort(bn_bias)
#     #     arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
#     #     output_channel_index = sorted(arg_max_rev.tolist())
#     #
#     #     pruned_channel_index = list(set(list(range(bn_weight.shape[0]))) - set(output_channel_index))
#     #
#     #     return output_channel_index, np.array(pruned_channel_index)
#
#     def get_decompose_weight(self):
#
#         # scale matrix
#         z = None
#
#         # copy original weight
#         self.decompose_weight = list(self.param_dict.values())
#
#         # cfg index
#         layer_id = -1
#
#         for index, layer in enumerate(self.param_dict):
#
#             original = self.param_dict[layer]
#
#             # VGG
#             if self.arch == 'ImageNet_VGG':
#
#                 # feature
#                 if 'feature' in layer:
#
#                     # conv
#                     if len(self.param_dict[layer].shape) == 4:
#
#                         layer_id += 1
#
#                         # get index
#                         self.output_channel_index[index], self.pruned_channel_index[
#                             index] = self.get_output_channel_index(self.param_dict[layer], layer_id)
#
#                         # Merge scale matrix
#                         if z != None:
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(self.param_dict[layer]):
#                                 o = f.view(f.shape[0], -1)
#                                 o = torch.mm(z, o)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#
#                         # make scale matrix with batchNorm
#                         bn = list(self.param_dict.values())
#
#                         bn_weight = bn[index + 1].cpu().detach().numpy()
#                         bn_bias = bn[index + 2].cpu().detach().numpy()
#                         bn_mean = bn[index + 3].cpu().detach().numpy()
#                         bn_var = bn[index + 4].cpu().detach().numpy()
#
#                         if self.model_type == 'merge' or self.model_type == 'prune':
#                             z = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                  self.threshold,
#                                                                  bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                                  self.model_type)
#                         elif self.model_type == 'OURS_PLUS':
#                             z, error_dict, alpha_dict = ours_create_scaling(
#                                 self.param_dict[layer].cpu().detach().numpy(),
#                                 np.array(self.output_channel_index[index]),
#                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.lamda_2,
#                                 self.compensation_threshold)
#                             self.error_dict[layer] = error_dict
#                             self.alpha_dict[layer] = alpha_dict
#
#                         z = torch.from_numpy(z).type(dtype=torch.float)
#                         if self.cuda:
#                             z = z.cuda()
#
#                         z = z.t()
#
#                         # pruned
#                         pruned = original[self.output_channel_index[index], :, :, :]
#                         # update next input channel
#                         input_channel_index = self.output_channel_index[index]
#
#                         # update decompose weight
#                         self.decompose_weight[index] = pruned
#
#
#                     # batchNorm
#                     elif len(self.param_dict[layer].shape):
#
#                         # pruned
#                         pruned = self.param_dict[layer][input_channel_index]
#
#                         # update decompose weight
#                         self.decompose_weight[index] = pruned
#
#
#                 # first classifier
#
#
#                 else:
#                     pruned = torch.zeros(original.shape[0], z.shape[0]*7*7) # 4096,12544
#
#                     if self.cuda:
#                         pruned = pruned.cuda()
#
#                     for i, f in enumerate(original): # original : 4096,25088
#                         o_old = f.view(z.shape[1], -1) # o_old = 512*49
#                         o = torch.mm(z, o_old).view(-1) # z = 256*512 --> o = 256*49 = 12544
#                         pruned[i, :] = o #
#                     self.decompose_weight[index] = pruned
#
#                     break
#
#             # ResNet
#             elif self.arch == 'ResNet':
#
#                 # block
#                 if 'layer' in layer:
#
#                     # last layer each block
#                     if '0.conv1.weight' in layer:
#                         layer_id += 1
#
#                     # Pruning
#                     if 'conv1' in layer:
#
#                         # get index
#                         self.output_channel_index[index], self.pruned_channel_index[
#                             index] = self.get_output_channel_index(self.param_dict[layer], layer_id)
#
#                         if self.model_type == 'merge':
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                  self.threshold,
#                                                                  bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                                  self.model_type)
#                             z = torch.from_numpy(x).type(dtype=torch.float)
#                             if self.cuda:
#                                 z = z.cuda()
#
#                             z = z.t()
#
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]
#                             # update next input channel
#                             input_channel_index = self.output_channel_index[index]
#
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'prune':
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'OURS_PLUS':
#
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             scale, error_dict, alpha_dict = ours_create_scaling(
#                                 self.param_dict[layer].cpu().detach().numpy(),
#                                 np.array(self.output_channel_index[index]),
#                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.lamda_2,
#                                 self.compensation_threshold)
#                             self.error_dict[layer] = error_dict
#                             self.alpha_dict[layer] = alpha_dict
#
#                             scale_mat = scale.T
#                             scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)
#
#                             if self.cuda:
#                                 scale_mat = scale_mat.cuda()
#
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#
#                         elif self.model_type == 'OURS':
#                             scaled_original = original.clone().detach()
#                             # for filter_idx in range(original.size(1)):
#                             #     similiar_dict = defaultdict(list)
#                             #     filter_channel_vec = original[:, filter_idx, :, :].view(original.size(0), -1).cpu()
#                             #
#                             #     sim_matrix = pairwise_distances(filter_channel_vec, metric="cosine")
#                             #
#                             #     sim_matrix[self.pruned_channel_index[index], :] = np.array(
#                             #         [float(0)] * sim_matrix.shape[0])
#                             #
#                             #     sim_matrix -= 1
#                             #
#                             #     for pruned_idx in self.pruned_channel_index[index]:
#                             #         similar_idx = np.argsort(sim_matrix[:, pruned_idx])[-1]
#                             #         # similar_idx = np.argsort(sim_matrix[:, pruned_idx])[len(self.pruned_channel_index[index])]
#                             #
#                             #         s_val = sim_matrix[:, pruned_idx][similar_idx]
#                             #         similiar_dict[similar_idx].append([pruned_idx, s_val])
#                             #
#                             #     res = {k: sorted(v, key=lambda x: x[1]) for k, v in similiar_dict.items()}
#                             #     calculated_dict = dict()
#                             #     #
#                             #     for key, value in res.items():
#                             #         calculated_dict[key] = value[-1]  # idx, similiarity
#                             #
#                             #     for preserved_idx, (pruned_idx , val) in calculated_dict.items():
#                             #         preserved_filter = original[preserved_idx, filter_idx, :, :].view(-1).cpu().numpy()
#                             #         pruned_filter = original[pruned_idx, filter_idx, :, :].view(-1).cpu().numpy()
#                             #
#                             #         preserved_norm = np.linalg.norm(preserved_filter, ord=2)
#                             #         pruned_norm = np.linalg.norm(pruned_filter, ord=2)
#                             #         scale = (preserved_norm / pruned_norm)
#                             #         # original[preserved_idx, filter_idx, :, :] *= scale
#                             #
#                             #         if 1 < scale < 1.6: # cosine similarity가 높아도, scale차이가 크면 continue
#                             #             scaled_original[preserved_idx, filter_idx, :, :] *= scale
#                             # # #
#                             # conv1_norm_dictionary = dict()
#                             # for i in self.output_channel_index[index]:
#                             #     ori_norm = np.linalg.norm(original[i,:,:,:].view(-1).cpu().detach().numpy(), ord=2)
#                             #     scaled_norm = np.linalg.norm(scaled_original[i,:,:,:].view(-1).cpu().detach().numpy(), ord=2)
#                             #     conv1_norm_dictionary[i] = float(scaled_norm/ori_norm)
#                             #     # scaled_original[i, :, :, :] *= float(scaled_norm/ori_norm)
#
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             self.decompose_weight[index] = pruned
#
#                     # batchNorm
#                     elif 'bn1' in layer:
#
#                         if len(self.param_dict[layer].shape):
#                             pruned = self.param_dict[layer][input_channel_index]
#                             # self.decompose_weight[index] = pruned
#
#                             if self.model_type == 'OURS':
#                                 # if 'weight' in layer:
#                                 #     # print(layer)
#                                 #     pruned = (torch.tensor(np.array(list(conv1_norm_dictionary.values()))) * pruned.cpu()).cuda()
#                                 # elif 'bias' in layer:
#                                 #     # print(layer)
#                                 #     pruned = (torch.tensor(np.array(list(conv1_norm_dictionary.values())))  * pruned.cpu()).cuda()
#                                 # elif 'running_var' in layer:
#                                 #     # print(layer)
#                                 #     pruned = (torch.tensor(np.array(list(conv1_norm_dictionary.values()))) * pruned.cpu()).cuda()
#                                 #
#                                 # elif 'running_mean' in layer:
#                                 #     # print(layer)
#                                 #     pruned = (torch.tensor(np.array(list(conv1_norm_dictionary.values()))) * pruned.cpu()).cuda()
#
#                                 self.decompose_weight[index] = pruned
#
#                             else:
#                                 self.decompose_weight[index] = pruned
#
#
#                     # Merge scale matrix
#                     elif 'conv2' in layer:
#
#                         if z != None:
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(
#                                     self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(z, o)  # (11,16) * (16, 9)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#                         elif self.model_type == 'prune':
#                             self.decompose_weight[index] = original[:, input_channel_index, :, :]
#
#                         elif self.model_type == 'OURS_PLUS':
#                             conv2_norm_dictionary = dict()
#                             scaled_original = original.clone().detach()
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(
#                                     self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
#                                 o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # for filter_idx in range(original.size(0)):
#                             #     original_filter_norm = torch.norm(scaled_original[filter_idx, :, :, :])
#                             #     preserved_filter_norm = torch.norm(original[filter_idx, :, :, :])
#                             #     scale = float(preserved_filter_norm / original_filter_norm)
#                             #     original[filter_idx, :, :, : ] *= scale
#                             #     conv2_norm_dictionary[filter_idx] = scale
#                             # self.decompose_weight[index] = original
#
#                             # # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#
#                         elif self.model_type == 'OURS':
#                             scaled_original = original.clone().detach()
#                             conv2_norm_dictionary = dict()
#
#                             # for input_idx in input_channel_index:
#                             #     scaled_original[:, input_idx, :, :] *= conv1_norm_dictionary[input_idx]
#
#                             """channels of filter norm"""
#                             for filter_idx in range(original.size(0)):
#                                 original_filter_norm = torch.norm(original[filter_idx, :, :, :])
#                                 preserved_filter_norm = torch.norm(original[filter_idx, input_channel_index, :, :])
#                                 scale = float(original_filter_norm / preserved_filter_norm)
#                                 original[filter_idx, :, :, :] *= scale
#                                 conv2_norm_dictionary[filter_idx] = scale
#                             self.decompose_weight[index] = original[:, input_channel_index, :, :]
#
#                             # """Pixel-wise norm"""
#                             # for filter_idx in range(original.size(0)):
#                             #     scale_matrix = list()
#                             #     original_filter = original[filter_idx, :, :, :].view(original.size(1), -1)
#                             #
#                             #     for kernel_idx in range(9):
#                             #         fixel_ori_norm = torch.norm(original_filter[:, kernel_idx], p = 2)
#                             #         fixel_pruned_norm = torch.norm(original_filter[input_channel_index, kernel_idx], p = 2)
#                             #         scale_matrix.append(float(fixel_ori_norm/fixel_pruned_norm))
#                             #
#                             #     mut_out =  original_filter * torch.tensor(scale_matrix).cuda()
#                             #
#                             #     scaled_original[filter_idx, :, :, : ] =  mut_out.view(mut_out.size(0), 3, 3)
#                             #
#                             #     for i in range(original.size(0)):
#                             #         ori_norm = np.linalg.norm(original[i, input_channel_index, :, :].view(-1).cpu().detach().numpy(),ord=2)
#                             #         scaled_norm = np.linalg.norm(scaled_original[i, input_channel_index, :, :].view(-1).cpu().detach().numpy(), ord=2)
#                             #         conv2_norm_dictionary[i] = float(scaled_norm / ori_norm)
#                             #
#                             # self.decompose_weight[index] = scaled_original[:, input_channel_index, :, :]
#
#                     elif 'bn2' in layer:
#                         if self.model_type == 'OURS':
#                             pruned = self.param_dict[layer]
#                             # # #
#                             # if 'weight' in layer:
#                             #     # print(layer)
#                             #     pruned = (torch.tensor(np.array(list(conv2_norm_dictionary.values()))) * pruned.cpu()).cuda()
#                             # elif 'bias' in layer:
#                             #     # print(layer)
#                             #     pruned = (torch.tensor(np.array(list(conv2_norm_dictionary.values())))  * pruned.cpu()).cuda()
#                             #
#                             # elif 'running_var' in layer:
#                             #     # print(layer)
#                             #     pruned = (torch.tensor(np.array(list(conv2_norm_dictionary.values()))) * pruned.cpu()).cuda()
#                             # #
#                             # elif 'running_mean' in layer:
#                             #     # print(layer)
#                             #     pruned = (torch.tensor(np.array(list(conv2_norm_dictionary.values()))) * pruned.cpu()).cuda()
#                             #
#                             self.decompose_weight[index] = pruned
#
#             elif self.arch == 'ImageNet_ResNet34':
#
#                 # block
#                 if 'layer' in layer:
#
#                     # last layer each block
#                     if '0.group1.conv1.weight' in layer:
#                         layer_id += 1
#
#                     # Pruning
#                     if 'conv1' in layer:
#
#                         # get index
#                         self.output_channel_index[index], self.pruned_channel_index[
#                             index] = self.get_output_channel_index(self.param_dict[layer], layer_id)
#
#                         if self.model_type == 'merge':
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                  self.threshold,
#                                                                  bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                                  self.model_type)
#                             z = torch.from_numpy(x).type(dtype=torch.float)
#                             if self.cuda:
#                                 z = z.cuda()
#
#                             z = z.t()
#
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]
#                             # update next input channel
#                             input_channel_index = self.output_channel_index[index]
#
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'prune':
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'OURS_PLUS':
#
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             scale, error_dict, alpha_dict = ours_create_scaling(
#                                 self.param_dict[layer].cpu().detach().numpy(),
#                                 np.array(self.output_channel_index[index]),
#                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.lamda_2,
#                                 self.compensation_threshold)
#                             self.error_dict[layer] = error_dict
#                             self.alpha_dict[layer] = alpha_dict
#
#                             scale_mat = scale.T
#                             scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)
#
#                             if self.cuda:
#                                 scale_mat = scale_mat.cuda()
#
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#
#                         elif self.model_type == 'OURS':
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             self.decompose_weight[index] = pruned
#
#                     # batchNorm
#                     elif 'bn1' in layer:
#
#                         if len(self.param_dict[layer].shape):
#                             pruned = self.param_dict[layer][input_channel_index]
#                             self.decompose_weight[index] = pruned
#
#
#                     # Merge scale matrix
#                     elif 'conv2' in layer:
#
#                         if z != None:
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(
#                                     self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(z, o)  # (11,16) * (16, 9)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#                         elif self.model_type == 'prune':
#                             self.decompose_weight[index] = original[:, input_channel_index, :, :]
#
#                         elif self.model_type == 'OURS_PLUS':
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(
#                                     self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
#                                 o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#                     elif 'bn2' in layer:
#                         if self.model_type == 'OURS':
#                             pruned = self.param_dict[layer]
#                             self.decompose_weight[index] = pruned
#
#
#             elif self.arch == 'ImageNet_ResNet101':
#
#                 # block
#                 if 'layer' in layer:
#
#                     # last layer each block
#                     if '.0.group1.conv1.weight' in layer:
#                         layer_id += 1
#
#                     # Pruning
#                     if 'conv1' in layer:
#
#                         # get index
#                         self.output_channel_index[index], self.pruned_channel_index[
#                             index] = self.get_output_channel_index(self.param_dict[layer], layer_id)
#
#                         if self.model_type == 'merge':
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                  self.threshold,
#                                                                  bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                                  self.model_type)
#                             z = torch.from_numpy(x).type(dtype=torch.float)
#                             if self.cuda:
#                                 z = z.cuda()
#
#                             z = z.t()
#
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]
#                             # update next input channel
#                             input_channel_index = self.output_channel_index[index]
#
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'prune':
#                             # pruned
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                         elif self.model_type == 'OURS_PLUS':
#
#                             bn = list(self.param_dict.values())
#
#                             bn_weight = bn[index + 1].cpu().detach().numpy()
#                             bn_bias = bn[index + 2].cpu().detach().numpy()
#                             bn_mean = bn[index + 3].cpu().detach().numpy()
#                             bn_var = bn[index + 4].cpu().detach().numpy()
#
#                             scale, error_dict, alpha_dict = ours_create_scaling(
#                                 self.param_dict[layer].cpu().detach().numpy(),
#                                 np.array(self.output_channel_index[index]),
#                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.lamda_2,
#                                 self.compensation_threshold)
#                             self.error_dict[layer] = error_dict
#                             self.alpha_dict[layer] = alpha_dict
#
#                             scale_mat = scale.T
#                             scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)
#
#                             if self.cuda:
#                                 scale_mat = scale_mat.cuda()
#
#                             pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
#                             input_channel_index = self.output_channel_index[index]
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#                     # batchNorm
#                     elif 'bn1' in layer:
#
#                         if len(self.param_dict[layer].shape):
#                             pruned = self.param_dict[layer][input_channel_index]
#                             self.decompose_weight[index] = pruned
#
#
#                     # Merge scale matrix
#                     elif 'conv2' in layer:
#                         # make scale matrix with batchNorm
#                         bn = list(self.param_dict.values())
#                         bn_weight = bn[index + 1].cpu().detach().numpy()
#                         bn_bias = bn[index + 2].cpu().detach().numpy()
#                         bn_mean = bn[index + 3].cpu().detach().numpy()
#                         bn_var = bn[index + 4].cpu().detach().numpy()
#                         self.output_channel_index[index], self.pruned_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)
#
#
#                         if z != None and self.model_type == 'merge':
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(z, o)  # (11,16) * (16, 9)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                  self.threshold,
#                                                                  bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                                  self.model_type)
#                             z = torch.from_numpy(x).type(dtype=torch.float)
#                             if self.cuda:
#                                 z = z.cuda()
#                             z = z.t()
#                             self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]
#
#
#                         elif self.model_type == 'prune':
#                             _original = original[:, input_channel_index, :, :]
#                             self.decompose_weight[index] = _original[self.output_channel_index[index], :, :, :]
#
#
#                         elif self.model_type == 'OURS_PLUS':
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
#                                 o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             scale, error_dict, alpha_dict = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
#                                                                  np.array(self.output_channel_index[index]),
#                                                                           bn_weight,bn_bias ,bn_mean,bn_var, self.lamda, self.lamda_2, self.compensation_threshold)
#                             self.error_dict[layer] = error_dict
#                             self.alpha_dict[layer] = alpha_dict
#
#                             scale_mat = scale.T
#                             scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)
#
#                             if self.cuda:
#                                 scale_mat = scale_mat.cuda()
#
#                             self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]
#
#                         input_channel_index = self.output_channel_index[index]
#
#
#                     elif 'bn2' in layer:
#                         if len(self.param_dict[layer].shape):
#                             pruned = self.param_dict[layer][input_channel_index]
#                             self.decompose_weight[index] = pruned
#
#                     elif 'conv3' in layer:
#
#                         if z != None:
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(z, o)  # (11,16) * (16, 9)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#                         elif self.model_type == 'prune':
#                             self.decompose_weight[index] = original[:, input_channel_index, :, :]
#
#                         elif self.model_type == 'OURS_PLUS':
#                             original = original[:, input_channel_index, :, :]
#                             for i, f in enumerate(
#                                     self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
#                                 o = f.view(f.shape[0], -1)  # 16,9
#                                 o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
#                                 o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#                             scaled = original
#
#                             # # update decompose weight
#                             self.decompose_weight[index] = scaled
#
#
#
#             # WideResNet
#             elif self.arch == 'WideResNet':
#
#                 # block
#                 if 'block' in layer:
#
#                     # last layer each block
#                     if '0.conv1.weight' in layer:
#                         layer_id += 1
#
#                     # Pruning
#                     if 'conv1' in layer:
#
#                         # get index
#                         self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer],
#                                                                                          layer_id)
#
#                         # make scale matrix with batchNorm
#                         bn = list(self.param_dict.values())
#
#                         bn_weight = bn[index + 1].cpu().detach().numpy()
#                         bn_bias = bn[index + 2].cpu().detach().numpy()
#                         bn_mean = bn[index + 3].cpu().detach().numpy()
#                         bn_var = bn[index + 4].cpu().detach().numpy()
#
#                         x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
#                                                              np.array(self.output_channel_index[index]), self.threshold,
#                                                              bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
#                                                              self.model_type)
#
#                         z = torch.from_numpy(x).type(dtype=torch.float)
#
#                         if self.cuda:
#                             z = z.cuda()
#
#                         z = z.t()
#
#                         # pruned
#                         pruned = original[self.output_channel_index[index], :, :, :]
#
#                         # update next input channel
#                         input_channel_index = self.output_channel_index[index]
#
#                         # update decompose weight
#                         self.decompose_weight[index] = pruned
#
#
#                     # BatchNorm
#                     elif 'bn2' in layer:
#
#                         if len(self.param_dict[layer].shape):
#                             # pruned
#                             pruned = self.param_dict[layer][input_channel_index]
#
#                             # update decompose weight
#                             self.decompose_weight[index] = pruned
#
#
#                     # Merge scale matrix
#                     elif 'conv2' in layer:
#
#                         # scale
#                         if z != None:
#                             original = original[:, input_channel_index, :, :]
#                             print(self.param_dict[layer])
#                             for i, f in enumerate(self.param_dict[layer]):
#                                 o = f.view(f.shape[0], -1)
#                                 o = torch.mm(z, o)
#                                 o = o.view(z.shape[0], f.shape[1], f.shape[2])
#                                 original[i, :, :, :] = o
#
#                         scaled = original
#
#                         # update decompose weight
#                         self.decompose_weight[index] = scaled
#
#             # LeNet_300_100
#             elif self.arch == 'LeNet_300_100':
#
#                 # ip
#                 if layer in ['ip1.weight', 'ip2.weight']:
#
#                     # Merge scale matrix
#                     if z != None:
#                         original = torch.mm(original, z)
#
#                     layer_id += 1
#
#                     # concatenate weight and bias
#                     if layer in 'ip1.weight':
#                         weight = self.param_dict['ip1.weight'].cpu().detach().numpy()
#                         bias = self.param_dict['ip1.bias'].cpu().detach().numpy()
#
#                     elif layer in 'ip2.weight':
#                         weight = self.param_dict['ip2.weight'].cpu().detach().numpy()
#                         bias = self.param_dict['ip2.bias'].cpu().detach().numpy()
#
#                     bias_reshaped = bias.reshape(bias.shape[0], -1)
#                     concat_weight = np.concatenate([weight, bias_reshaped], axis=1)
#
#                     # get index
#                     self.output_channel_index[index] = self.get_output_channel_index(torch.from_numpy(concat_weight),
#                                                                                      layer_id)
#
#                     # make scale matrix with bias
#                     x = create_scaling_mat_ip_thres_bias(concat_weight, np.array(self.output_channel_index[index]),
#                                                          self.threshold, self.model_type)
#                     z = torch.from_numpy(x).type(dtype=torch.float)
#
#                     if self.cuda:
#                         z = z.cuda()
#
#                     # pruned
#                     pruned = original[self.output_channel_index[index], :]
#
#                     # update next input channel
#                     input_channel_index = self.output_channel_index[index]
#
#                     # update decompose weight
#                     self.decompose_weight[index] = pruned
#
#                 elif layer in 'ip3.weight':
#
#                     original = torch.mm(original, z)
#
#                     # update decompose weight
#                     self.decompose_weight[index] = original
#
#                 # update bias
#                 elif layer in ['ip1.bias', 'ip2.bias']:
#                     self.decompose_weight[index] = original[input_channel_index]
#
#                 else:
#                     pass
#
#     def main(self):
#
#         if self.cuda == False:
#             for layer in self.param_dict:
#                 self.param_dict[layer] = self.param_dict[layer].cpu()
#
#         self.get_decompose_weight()
#
#         return self.decompose_weight, self.error_dict, self.alpha_dict
