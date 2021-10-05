import torch
from torch.tensor import _wrap_type_error_to_not_implemented

def check_collisison():
    NotImplementedError

def step_cars(sim) -> torch.Tensor:
    #https://www.cs.cmu.edu/~motionplanning/reading/PlanningforDynamicVeh-1.pdf
    # 
    # 
    #blah wip


    state = sim.state

    return state