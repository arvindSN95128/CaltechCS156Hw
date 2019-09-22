"""
Implementation of Perceptron learning Algorithm

In this problem, you will create your own target function f and data set D to see
how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
problem, and assume X = [−1, 1] × [−1, 1] with uniform probability of picking each
x ∈ X .
In each run, choose a random line in the plane as your target function f (do this by
taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
line passing through them), where one side of the line maps to +1 and the other maps
to −1. Choose the inputs xn of the data set as random points (uniformly in X ), and
evaluate the target function on each xn to get the corresponding output yn.
Now, in each run, use the Perceptron Learning Algorithm to find g. Start the PLA
with the weight vector w being all zeros (consider sign(0) = 0, so all points are initially misclassified), and at each iteration have the algorithm choose a point randomly
from the set of misclassified points. 

We are interested in two quantities: 
    1. the number of iterations that PLA takes to converge to g, 
    and 
    2. the disagreement between f and g which is P[f(x) != g(x)] (the probability that f and g will disagree on their classification of a random point). You can either calculate this probability exactly, or
approximate it by generating a sufficiently large, separate set of points to estimate it.

In order to get a reliable estimate for these two quantities, you should repeat the
experiment for 1000 runs (each run as specified above) and take the average over
these runs.

"""
import numpy as np
import random
import collections

#print('Inverse of\n'  + repr(np.array([[2, 2], [3, 1]])) + '\nis\n' + repr(np.linalg.inv(np.array([[2, 2], [3, 1]]))))

def execute_pla(iter_number, num_samples, start_with_regression_weigths = False):
    """
    Step 1: Choose a random line in the plane as your target function f (do this by
    taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
    line passing through them), where one side of the line maps to +1 and the other maps
    to −1.
    """
    random.seed(a = iter_number)
    (x1, y1) = (random.uniform(-1, 1), random.uniform(-1, 1))
    (x2, y2) = (random.uniform(-1, 1), random.uniform(-1, 1))

    m = (y2 - y1) / (x2 - x1) if (x2 - x1 != 0) else float("inf")
    c = y1 - m * x1 if (m != float("inf")) else float("inf")
    #print("(x1, y1) = " + repr((x1, y1)) + ", (x2, y2) = " + repr((x2, y2)), " line is y = %.3f * x + %.3f" % (m, c))
    

    """
    Step 2: Choose the inputs xn of the data set as random points (uniformly in X ), and 
    evaluate the target function on each xn to get the corresponding output yn.
    """
    # The side with (1/2, 1/2) is +1
    ref_sign = np.sign(m * 0.5 - 0.5 + c)

    Samples = collections.namedtuple('Samples', ['input', 'output'])
    samples = []
    for sample_ind in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        output = 1 if np.sign(m * x - y + c) ==  ref_sign else -1
        samples.append(Samples(np.array([1, x, y]), output))


    #print("samples are" + repr(samples))
    '''
    Now, in each run, use the Perceptron Learning Algorithm to find g. Start the PLA with the weight vector w being all
    zeros (consider sign(0) = 0, so all points are initially misclassified), and at each iteration have the algorithm
    choose a point randomly from the set of misclassified points. 
    
    We are interested in two quantities: the number of iterations that PLA takes to converge to g, and the disagreement
    between f and g which is P[f(x) != g(x)] (the probability that f and g will disagree on their classification of a
    random point). You can either calculate this probability exactly, or approximate it by generating a sufficiently large,
    separate set of points to estimate it.
    '''
    weights = np.array([0, 0, 0])
    num_iter = 0
    num_misclassified  = num_samples

    while num_iter <= 1e8:
        num_iter += 1
        ## get indices of all misclassified samples
        num_misclassified = 0
        misclassified  = []
        for samp_index in range(num_samples):
            if (np.dot(weights, samples[samp_index].input) * samples[samp_index].output <= 0):
                num_misclassified += 1
                misclassified.append(samp_index)

        if num_misclassified == 0:
            break

        # adjust weights to classify 1 random misclassified point
        rand_sample = random.randint(0, num_misclassified - 1)
        weights = weights + samples[misclassified[rand_sample]].output * samples[misclassified[rand_sample]].input
    ##end while

    #print("Target line is y = %.3f * x + %.3f. Num iter is %d" % (m, c, num_iter))
    #print("Predicted line is y = %.3f x + %.3f" % (-1.0 * weights[1] / weights[2] , -1.0 * weights[0] / weights[2]))

    #find prob or error
    num_error = 0
    num_samples_to_calc_prob = 10000
    for i in range(num_samples_to_calc_prob):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        target_val = 1 if np.sign(m * x - y + c) == ref_sign else -1
        hyp_val  = 1 if np.dot(weights, np.array([1, x, y])) > 0 else -1
        if (target_val != hyp_val):
            num_error += 1
    return (num_iter, num_error * 1.0 / num_samples_to_calc_prob)



ave_num_iter_converge = 0.0
ave_p_error = 0.0
num_runs = 1000
num_samples = 100
'''
In order to get a reliable estimate for these two quantities, you should repeat
    the experiment for 1000 runs (each run as specified above) and take the average over these runs.
'''
for i in range(num_runs):
    (num_iter_converge, p_error) = execute_pla(i, num_samples)
    ave_p_error +=  p_error
    ave_num_iter_converge += num_iter_converge


print("Num samples : %d, Ave Prob error is %.3f and ave num iters is %.3f" % (num_samples, ave_p_error / num_runs, ave_num_iter_converge / num_runs))

