import torch
import numpy as np
from torch.autograd import Variable


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