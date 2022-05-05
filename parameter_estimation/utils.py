import torch
import numpy as np
from torch.autograd import Variable

import torch.optim as optim


def mapfit(pdf, prior, parameters, observations, iter=1000, lr=0.1):
    """Estimates the parameters of an arbitrary function via maximum a posteriori estimation and
    uses plain old gradient descent for optimization
    Parameters
    ----------
    func :          Callable pdf
                    Callable probability density function (likelihood function)
                    expecting an array of observations as the only argument.
                    e.g. p(x|params) = func(observations)
    prior :         Callable pdf
                    Callable probability density function over parameters
                    expecting an array of parameters as the only argument.
                    e.g. p(params) = prior(parameters)
                    if p(params) is a uniform distribution, this method equals MLE
    parameters :    List
                    List of parameters that are subject to optimization.
    observations :  ndarray
                    Observations from an unknown pdf which parameters are subject to be estimated
    iter :          float
                    Maximum number of iterations
    lr :            float
                    Gradient descent learning rate
    Returns
    -------
    """

    for i in range(iter):
        # Define objective function (log-likelihood) to maximize
        prior_ = torch.log(prior(parameters))
        posterior = torch.mean(torch.log(pdf(*observations))) + prior_
        posterior = -1.0 * posterior

        if np.isnan(posterior.data[0]) or np.isnan(prior_.data[0]):
            return

        # Determine gradients
        posterior.backward()

        # Update parameters with gradient descent
        for param in parameters:
            param.data.add_(lr * param.grad.data)
            param.grad.data.zero_()


def mlefit(func, parameters, observations, iter=1000, lr=0.1):
    """Estimates the parameters of an arbitrary function via maximum likelihood estimation and
    uses plain old gradient descent for optimization
    Parameters
    ----------
    func :          Callable pdf
                    Callable probability density function (likelihood function)
                    expecting an array of observations as the only argument.
    parameters :    List
                    List of parameters that are subject to optimization.
    observations :  ndarray
                    Observations from an unknown pdf which parameters are subject to be estimated
    iter :          float
                    Maximum number of iterations
    lr :            float
                    Gradient descent learning rate
    Returns
    -------
    """

    # Use MAP with uniform prior
    prior_ = Variable(torch.tensor(1.0))
    return mapfit(func, lambda x: prior_, parameters, observations, iter, lr)

    # Explicit implementation without prior:
    # for i in range(iter):
    #     # Define objective function (log-likelihood) to maximize
    #     likelihood = torch.mean(torch.log(func(observations)))
    #
    #     # Determine gradients
    #     likelihood.backward()
    #
    #     # Update parameters with gradient descent
    #     for param in parameters:
    #         param.data.add_(lr * param.grad.data)
    #         param.grad.data.zero_()



def forcefit(dynamics, parameters, observations, iter=1000, lr=1e-3):
    """Estimates the parameters of an arbitrary function via maximum a posteriori estimation and
    uses plain old gradient descent for optimization
    Parameters
    ----------
    func :          Callable pdf
                    Callable probability density function (likelihood function)
                    expecting an array of observations as the only argument.
                    e.g. p(x|params) = func(observations)
    prior :         Callable pdf
                    Callable probability density function over parameters
                    expecting an array of parameters as the only argument.
                    e.g. p(params) = prior(parameters)
                    if p(params) is a uniform distribution, this method equals MLE
    parameters :    List
                    List of parameters that are subject to optimization.
    observations :  ndarray
                    Observations from an unknown pdf which parameters are subject to be estimated
    iter :          float
                    Maximum number of iterations
    lr :            float
                    Gradient descent learning rate
    Returns
    -------
    """

    optimizer = optim.Adam(parameters, lr=lr)

    states_tm1 = observations[0]
    controls_tm1 = observations[1]
    states_t = observations[2]
    dtime = observations[3]

    batch_size = 1024

    loss_fc = torch.nn.L1Loss()
    

    for i in range(iter):
        indices = torch.randperm(states_tm1.shape[0], requires_grad=False)[:batch_size]

        s_tm1 = states_tm1[indices]
        a_tm1 = controls_tm1[indices]
        s_t = states_t[indices]
        dt = dtime[indices]

        # Define objective function (log-likelihood) to maximize
        predicted = dynamics(s_tm1, a_tm1, s_t, dt)
        loss = loss_fc(predicted, s_t)

        # Determine gradients
        # loss.backward(retain_graph=True)

        # # Update parameters with gradient descent
        # for param in parameters:
        #     param.data.add_(lr * param.grad.data)
        #     param.grad.data.zero_()

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        optimizer.step()
        print(loss)

    return parameters