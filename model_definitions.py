import pickle
import os
import numpy
import pdb

import matplotlib.pyplot as plt
from scipy.special import gammaincc
from scanf import scanf


def load_in_data(filename, GL_min=0, GL_max=numpy.inf):
  with open(filename, 'rb') as pickle_file:
    data = pickle.load(pickle_file, encoding='latin1')

    Bbar_s = []
    N_s = []
    g_s = []
    L_s = []
    m_s = []

    # Array to hold all of the data
    samples = []

    for key in data:
      if key[:16] == "DBinder_crossing":
        # print(key)

        # Extract the parameter values
        Bbar, N, g, L = scanf("DBinder_crossing_B%f_%d_%f_%d", key)

        Bbar_s.append(Bbar)
        N_s.append(N)
        g_s.append(g)
        L_s.append(L)
        
        # Extract the observed mass value
        m_s.append(data[key][4][0])

        # Now extract the 500 bootstrap samples
        samples.append(data[key][4][2])

    samples = numpy.array(samples)

  # Turn data into numpy arrays
  N_s = numpy.array(N_s)
  g_s = numpy.array(g_s)
  L_s = numpy.array(L_s)
  Bbar_s = numpy.array(Bbar_s)
  m_s = numpy.array(m_s)

  # Remove nan values
  keep = numpy.logical_not(numpy.isnan(samples))[:, 0]
  samples = samples[keep]
  N_s = N_s[keep]
  g_s = g_s[keep]
  L_s = L_s[keep]
  Bbar_s = Bbar_s[keep]
  m_s = m_s[keep]

  GL_s = g_s * L_s

  # Check that all data has the same N
  assert len(set(N_s)) == 1

  # Apply a cut to the data
  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10),
                           GL_s <= GL_max * (1 + 10 ** -10))

  return samples[keep], g_s[keep], L_s[keep], Bbar_s[keep], m_s[keep]


# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  Z0 = 0.252731
  return - g * Z0 * (2 - 3 / N ** 2)


# Calculate the lambda terms
def K1(g, N):
  return numpy.log((g / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
  return numpy.log(1 / L) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


# No corrections to scaling model
def model1_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  """Lambda_IR proportional to g, with a single alpha coefficient"""
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K1(g, N))


def model2_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  """Lambda_IR proportional to L, with a single alpha coefficient"""
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K2(L, N))


def model1_2a(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
  """Lambda_IR proportional to g, with two alpha coefficients"""
  Bbar_s = numpy.sort(list(set(Bbar)))
  alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K1(g, N))


def model2_2a(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
  """Lambda_IR proportional to L, with two alpha coefficients"""
  Bbar_s = numpy.sort(list(set(Bbar)))
  alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K2(L, N))


def cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut):
  # In reality the covariance between different ensembles is 0. We can set it as
  # such to get a more accurate calculation of the covariance matrix
  different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

  # Check if two ensembles have different physical parameters, e.g. they are different
  # ensembles
  for i in range(samples_cut.shape[0]):
    for j in range(samples_cut.shape[0]):
      different_g[i, j] = g_s_cut[i] != g_s_cut[j]
      different_L[i, j] = L_s_cut[i] != L_s_cut[j]

  # This is true if two data points come different simulations
  different_ensemble = numpy.logical_or(different_L, different_g)

  # Find the covariance matrix - method 1 - don't enforce independence between ensembles
  # Calculate by hand so that the means from 
  size = samples_cut.shape[0]
  cov_matrix = numpy.zeros((size, size))
  for i in range(size):
    for j in range(size):
      if different_ensemble[i, j] == 0:
        cov_matrix[i, j] = numpy.mean((samples_cut[i] - m_s_cut[i]) * (samples_cut[j] - m_s_cut[j]))
      # cov_matrix[i, j] = numpy.mean((samples_cut[i] - numpy.mean(samples_cut[i])) * (samples_cut[j] - numpy.mean(samples_cut[j])))
      # else the value remains zero as there is no covariance between samples from different ensembles

  return cov_matrix, different_ensemble


def chisq_calc(x, cov_inv, model_function, res_function):
  # Caculate the residuals between the model and the data
  normalized_residuals = res_function(x, cov_inv, model_function)

  chisq = numpy.sum(normalized_residuals ** 2)

  return chisq


def chisq_pvalue(k, x):
  "k is the rank, x is the chi-sq value"
  return gammaincc(k / 2, x / 2)


# Try using the scipy least-squares method with Nelder-Mead
def make_res_function(N, m_s, g_s, L_s, Bbar_s):
  def res_function(x, cov_inv, model_function):

    # Caculate the residuals between the model and the data
    predictions = model_function(N, g_s, L_s, Bbar_s, *x)

    residuals = m_s - predictions

    normalized_residuals = numpy.dot(cov_inv, residuals)

    return normalized_residuals

  return res_function
