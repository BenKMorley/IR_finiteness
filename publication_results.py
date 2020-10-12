from frequentist_run import run_frequentist_analysis
from model_definitions import *
from tqdm import tqdm
from bayesian_run import *


## Part I : Producing p-values for figure 4.
def part1():
  run_bootstrap = False
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  # N = 2
  N = 2
  Bbar_1 = "0.520"
  Bbar_2 = "0.530"
  x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values

  model1 = model1_1a
  model2 = model2_1a
  param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

  pvalues_1 = numpy.zeros(len(GL_mins))
  pvalues_2 = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    pvalues_1[i], params1, dof = run_frequentist_analysis(model1, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False)
    pvalues_2[i], params2, dof = run_frequentist_analysis(model2, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False)

  # N = 4
  N = 4
  Bbar_1 = "0.420"
  Bbar_2 = "0.430"
  x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values

  model1 = model1_2a
  model2 = model2_2a
  param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

  pvalues_1 = numpy.zeros(len(GL_mins))
  pvalues_2 = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    pvalues_1[i], params1, dof = run_frequentist_analysis(model1, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False)
    pvalues_2[i], params2, dof = run_frequentist_analysis(model2, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False)


## Part II: Producing parameter estimates and statistical error for central fit
def part2():
  # N = 2
  model = model1_1a
  N = 2
  GL_min = 12.8
  GL_max = 76.8

  Bbar_1 = "0.520"
  Bbar_2 = "0.530"
  x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values
  param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

  pvalue, params1, dof, sigmas = run_frequentist_analysis(model, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0)


  # N = 4
  model = model1_2a
  N = 4
  GL_min = 12.8
  GL_max = 76.8

  Bbar_1 = "0.420"
  Bbar_2 = "0.430"
  x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values
  param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

  pvalue, params1, dof, sigmas = run_frequentist_analysis(model, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0)


## PART III: Calculating a systematic error based on different Bbar combinations
def part3():
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  # N = 2
  N = 2
  model = model1_1a
  min_dof = 15

  x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values
  param_names = ["alpha", "f0", "f1", "lambduh", "nu"]
  n_params = len(param_names)
  Bbar_s = ["0.510", "0.520", "0.530", "0.540", "0.550", "0.560", "0.570", "0.580", "0.590"]

  Bbar_list = []
  for i in range(len(Bbar_s)):
    for j in range(i + 1, len(Bbar_s)):
      Bbar_list.append([Bbar_s[i], Bbar_s[j]])

  pvalues = numpy.zeros((len(Bbar_list), len(GL_mins)))
  params = numpy.zeros((len(Bbar_list), len(GL_mins), n_params))
  dofs = numpy.zeros((len(Bbar_list), len(GL_mins)))

  for i, Bbar_s in enumerate(Bbar_list):
    Bbar_1, Bbar_2 = Bbar_s
    print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")

    for j, GL_min in enumerate(GL_mins):
      pvalues[i, j], params[i, j], dofs[i, j] = run_frequentist_analysis(model, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False, print_info=False)

  r = len(GL_mins)
  best = r - 1

  for i, GL_min in enumerate(GL_mins):
    if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
      best = r - 1 - i

  best_Bbar_index = numpy.argmax(pvalues[:, best])
  best_Bbar = Bbar_list[best_Bbar_index]

  print("BEST RESULT")
  print(f"Bbar_s = {best_Bbar}")
  print(f"GL_min = {GL_mins[best]}")
  print(f"pvalue : {pvalues[best_Bbar_index, best]}")
  print(f"dof : {dofs[best_Bbar_index, best]}")

  ### GET THE SYSTEMATIC FITS
  acceptable = numpy.logical_and(
               numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                      dofs >= min_dof)

  for i, param in enumerate(param_names):
    param_small = params[..., i]
    minimum = numpy.min(param_small[acceptable])
    maximum = numpy.max(param_small[acceptable])

  for i, param in enumerate(param_names):
    sys_sigma = max(maximum - params[best_Bbar_index, best, i], params[best_Bbar_index, best, i] - minimum)

    print(f"{param} = {param} +- {sys_sigma}")

  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  # N = 4
  N = 4
  model = model1_2a
  min_dof = 15

  x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values
  param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]
  n_params = len(param_names)
  Bbar_s = ["0.420", "0.430", "0.440", "0.450", "0.460", "0.470"]

  Bbar_list = []
  for i in range(len(Bbar_s)):
    for j in range(i + 1, len(Bbar_s)):
      Bbar_list.append([Bbar_s[i], Bbar_s[j]])

  pvalues = numpy.zeros((len(Bbar_list), len(GL_mins)))
  params = numpy.zeros((len(Bbar_list), len(GL_mins), n_params))
  dofs = numpy.zeros((len(Bbar_list), len(GL_mins)))

  for i, Bbar_s in enumerate(Bbar_list):
    Bbar_1, Bbar_2 = Bbar_s
    print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")

    for j, GL_min in enumerate(GL_mins):
      pvalues[i, j], params[i, j], dofs[i, j] = run_frequentist_analysis(model, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, run_bootstrap=False, print_info=False)

  r = len(GL_mins)
  best = r - 1

  for i, GL_min in enumerate(GL_mins):
    if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
      best = r - 1 - i

  best_Bbar_index = numpy.argmax(pvalues[:, best])
  best_Bbar = Bbar_list[best_Bbar_index]

  print("BEST RESULT")
  print(f"Bbar_s = {best_Bbar}")
  print(f"GL_min = {GL_mins[best]}")
  print(f"pvalue : {pvalues[best_Bbar_index, best]}")
  print(f"dof : {dofs[best_Bbar_index, best]}")

  ### GET THE SYSTEMATIC FITS
  acceptable = numpy.logical_and(
               numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                      dofs >= min_dof)

  for i, param in enumerate(param_names):
    param_small = params[..., i]
    minimum = numpy.min(param_small[acceptable])
    maximum = numpy.max(param_small[acceptable])

  for i, param in enumerate(param_names):
    sys_sigma = max(maximum - params[best_Bbar_index, best, i], params[best_Bbar_index, best, i] - minimum)

    print(f"{param} = {param} +- {sys_sigma}")


## Part IV: Calculating the Bayes Factor for figure 4.
def part4():
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  # N = 2
  N = 2
  Bbar_1 = "0.520"
  Bbar_2 = "0.530"
  x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values

  model1 = model1_1a
  model2 = model2_1a
  param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

  alpha_range = [-0.1, 0.1]
  f0_range = [0, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0, 2]
  prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]

  Bayes_factors2 = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'../../input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

    dof = g_s.shape[0] - n_params

    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    analysis1, best_fit1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True)

    analysis2, best_fit2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True)

    Bayes_factors2[i] = analysis1[0] - analysis2[0]

  # N = 4
  N = 4
  Bbar_1 = "0.420"
  Bbar_2 = "0.430"
  x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values

  model1 = model1_2a
  model2 = model2_2a
  param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

  alpha_range = [-0.1, 0.1]
  f0_range = [0, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0, 2]
  prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]

  Bayes_factors4 = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'../../input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

    dof = g_s.shape[0] - n_params

    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    analysis1, best_fit1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True)

    analysis2, best_fit2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True)

    Bayes_factors4[i] = analysis1[0] - analysis2[0]


part4()
