"""
Implementation of Non linear transformation

"""
import numpy as np
import random


# print('Inverse of\n'  + repr(np.array([[2, 2], [3, 1]])) + '\nis\n' + repr(np.linalg.inv(np.array([[2, 2], [3, 1]]))))


def execute_lin_reg(num_samples):
    '''
       Generate a training set of N = 1000 points on X = [−1, 1] × [−1, 1] with a uniform
       probability of picking each x ∈ X . Generate simulated noise by flipping the sign of
       the output in a randomly selected 10% subset of the generated training set.
       '''
    X = np.empty([num_samples, 3])
    X_non_lin = np.empty([num_samples, 6])
    Y = np.empty([num_samples, 1])
    for sample_ind in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        X[sample_ind] = [1, x, y]
        X_non_lin[sample_ind] = [1, x, y, x * y, x * x, y * y]
        f = 1 if np.sign(x * x + y * y - 0.6) >= 0 else -1
        Y[sample_ind] = [f] if random.uniform(0, 1) >= 0.1 else [-f]

    weights = np.dot(np.linalg.pinv(X), Y).T
    weights_non_lin = np.dot(np.linalg.pinv(X_non_lin), Y).T
    num_errors_lin = 0
    num_errors_non_lin = 0
    # find E_in with regression weights
    for sample_ind in range(num_samples):
        if np.sum(np.dot(weights, X[sample_ind])) * np.sum(Y[sample_ind]) <= 0:
            num_errors_lin += 1
        if np.sum(np.dot(weights_non_lin, X_non_lin[sample_ind])) * np.sum(Y[sample_ind]) <= 0:
            num_errors_non_lin += 1
    #print("E_in with linear regression is %.3f" % (num_errors * 1.0 / num_samples))

    num_e_out_lin = 0
    num_e_out_non_lin = 0
    num_samples_to_calc_prob = 1000
    for ind in range(num_samples_to_calc_prob):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        target_val = 1 if np.sign(x * x + y * y - 0.6) >= 0 else -1
        target_val = target_val if random.uniform(0, 1) >= 0.1 else -target_val
        hyp_val_lin = 1 if np.sum(np.dot(weights, np.array([1, x, y]))) > 0 else -1
        hyp_val_non_lin = 1 if np.sum(np.dot(weights_non_lin, np.array([1, x, y, x * y, x * x, y * y]))) > 0 else -1
        if target_val != hyp_val_lin:
            num_e_out_lin += 1
        if target_val != hyp_val_non_lin:
            num_e_out_non_lin += 1

    return (num_errors_lin * 1.0 / num_samples, num_e_out_lin * 1.0 / num_samples_to_calc_prob, num_errors_non_lin * 1.0 / num_samples, num_e_out_non_lin * 1.0 / num_samples_to_calc_prob, weights_non_lin)


ave_e_in_lin = 0.0
ave_e_out_lin = 0.0
ave_e_in_non = 0.0
ave_e_out_non = 0.0
ave_weights_non_lin = np.zeros([1, 6])
num_runs = 1000
num_samples = 1000
'''
    In order to get a reliable estimate for these two quantities, you should repeat
        the experiment for 1000 runs (each run as specified above) and take the average over these runs.
'''
for i in range(num_runs):
    (e_in_lin, e_out_lin, e_in_nonlin, e_out_nonlin, non_lin_feature_weights) = execute_lin_reg(num_samples)
    ave_e_in_lin += e_in_lin
    ave_e_out_lin += e_out_lin
    ave_e_in_non += e_in_nonlin
    ave_e_out_non += e_out_nonlin
    ave_weights_non_lin = non_lin_feature_weights + ave_weights_non_lin

print("Weights are")
for i in range(6):
    print("%d: %.4f" % (i, ave_weights_non_lin[0, i] / num_runs))

print("Lin feature: E_in = %.3f, E_out = %.3f\nNon lin feature: E_in = %.3f, E_out = %.3f" % (ave_e_in_lin / num_runs, ave_e_out_lin / num_runs, ave_e_in_non / num_runs, ave_e_out_non / num_runs))

