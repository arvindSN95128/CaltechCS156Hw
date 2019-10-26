import numpy as np
import random

# Run a computer simulation for flipping 1,000 virtual fair coins. Flip each coin inde-pendently 10 times.
# Focus on 3 coins as follows: c1 is the first coin flipped, crand is a coin chosen randomly from the 1,000, and cmin is the coin which had the minimum frequency of heads (pick the earlier one in case of a tie).
# Let ν1, νrand, and νmin be the fraction of heads obtained for the 3 respective coins out of the 10 tosses.
# Run the experiment 100,000 times in order to get a full distribution of ν1, νrand, and νmin (note that crand and cmin will change from run to run).

v_1 = v_rand = v_min = 0.0
num_exps = 100000
for exp_iter in range(0, num_exps):
    #random.seed
    num_coins = 1000
    num_flips = 10
    rand_coin_index = random.randint(0, num_coins - 1)
    min_v_this_iter = num_flips + 1
    print("Iter = %d Min = %.2f, 1 = %.2f, Rand = %.2f" % (exp_iter, v_min, v_1, v_rand))
    for coin_ind in range(0, num_coins):
        curr_coin_v = 0
        for flip_num in range(0, num_flips):
            if (random.random() > 0.5):
                curr_coin_v += 1


        min_v_this_iter = curr_coin_v if (curr_coin_v < min_v_this_iter) else min_v_this_iter

        if coin_ind == 1:
            v_1 += (curr_coin_v / num_flips)

        if coin_ind == rand_coin_index:+
            v_rand += (curr_coin_v / num_flips)

    v_min += (min_v_this_iter / num_flips)

v_1 = v_1 / num_exps
v_min = v_min / num_exps
v_rand = v_rand / num_exps

print("Min = %.2f, 1 = %.2f, Rand = %.2f" % (v_min, v_1, v_rand))
