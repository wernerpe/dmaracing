import trueskill
import numpy as np
from trueskill import Rating

r1 = Rating()
r2 = Rating()
r3 = Rating()

groups = [(r1,), (r2,), (r3,)]

newratings = trueskill.rate(groups, ranks = [2, 0, 1], weights={(0,0):0.2, (1,0):0.2, (2,0):0.2})

print('test')