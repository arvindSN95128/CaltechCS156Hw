"""
Implementation of Non linear transformation

"""
import numpy as np
import random


# print('Inverse of\n'  + repr(np.array([[2, 2], [3, 1]])) + '\nis\n' + repr(np.linalg.inv(np.array([[2, 2], [3, 1]]))))

def gen_training_samples(num_samples):


    return (X, Y)


def execute_lin_reg_lin_feature(num_samples):
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

    return (num_errors_lin * 1.0 / num_samples, num_errors_non_lin * 1.0 / num_samples, weights_non_lin)


ave_p_error_lin_feature = 0.0
ave_p_error_non_lin_feature = 0.0
ave_weights_non_lin = np.empty([1, 6])
num_runs = 1000
num_samples = 1000
'''
    In order to get a reliable estimate for these two quantities, you should repeat
        the experiment for 1000 runs (each run as specified above) and take the average over these runs.
'''
for i in range(num_runs):
    (p_error, p_error_non_lin, non_lin_feature_weights) = execute_lin_reg_lin_feature(num_samples)
    ave_p_error_lin_feature += p_error
    ave_p_error_non_lin_feature += p_error_non_lin
    ave_weights_non_lin = non_lin_feature_weights + ave_weights_non_lin

print("Weights are")
for i in range(6):
    print("%d: %.4f" % (i, ave_weights_non_lin[0, i] / num_runs))

print("Ave Prob error with lin features is %.3f, With Non-lin features is %.3f" % (ave_p_error_lin_feature / num_runs, ave_p_error_non_lin_feature / num_runs))

