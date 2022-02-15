import numpy as np

# avgranking = np.array([1.5, 2.8, 2.3, 2.7, 5])
# env_of_rank = np.argsort(avgranking)

# ranks_final = 0*env_of_rank
# for idx, env in enumerate(env_of_rank[1:]):
#     #avg(env (rank i))- avg(env (rank i-1))>eps 
#     if avgranking[env]- avgranking[env_of_rank[idx]]  > 0.2:
#         ranks_final[env] = ranks_final[env_of_rank[idx]] + 1 
#     else:
#         ranks_final[env] = ranks_final[env_of_rank[idx]]
# ranks_final = ranks_final.tolist()
# print('done')

import trueskill
import matplotlib.pyplot as plt

result1 = [0,1,0,1]
result2 = [1,0,1,0]
start = [5, 10, 15, 20]
N = 1000

ratings = [(trueskill.Rating(mu = s),) for s in start]

mu = [start]
for it in range(N):
    if it%1000 ==0:
        print(it)
    if int(it/25) %2 == 0:
        result = result1
    else:
        result = result2
    ratings = trueskill.rate(ratings, result)
    mu.append([ratings[idx][0].mu for idx in range(len(ratings))])

plt.figure()
plt.plot(np.array(mu))
plt.show()