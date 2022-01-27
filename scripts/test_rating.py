import numpy as np

avgranking = np.array([1.5, 2.8, 2.3, 2.7, 5])
env_of_rank = np.argsort(avgranking)

ranks_final = 0*env_of_rank
for idx, env in enumerate(env_of_rank[1:]):
    #avg(env (rank i))- avg(env (rank i-1))>eps 
    if avgranking[env]- avgranking[env_of_rank[idx]]  > 0.2:
        ranks_final[env] = ranks_final[env_of_rank[idx]] + 1 
    else:
        ranks_final[env] = ranks_final[env_of_rank[idx]]
ranks_final = ranks_final.tolist()
print('done')
