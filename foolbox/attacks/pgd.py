from __future__ import division
import numpy as np

from .base import Attack


class ProjectedGradientAttack(Attack):
    """L infinity Projected gradient descent attack from Madry et al. 2017.
    https://arxiv.org/abs/1706.06083

    Only implements xentropy loss for now. Foolbox does not implement CW attack
    Uses parameters from paper for MNIST:
    steps: 40,
    step size: 0.01
    epsilon: 0.3
    random start: true

    """

    def _apply(self, a, epsilon=.3, steps=40, lr=0.01):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()
        # random
        perturbed = image
        # perturbed = perturbed + np.random.uniform(-epsilon, epsilon, image.shape)
        for i in range(steps):
            gradient = a.gradient(perturbed)
            perturbed_sign = lr * np.sign(gradient)
            perturbed += perturbed_sign
            perturbed_epsilon = np.clip(perturbed, image-epsilon, image+epsilon)
            perturbed = np.clip(perturbed_epsilon, min_, max_)
            # try:
            a.predictions(perturbed)
            """
            except AssertionError:
                print "bounds"
                print i
                print perturbed.shape
                print np.min(perturbed), np.max(perturbed)
                print "adv: ", perturbed
                print "adv sign: ", perturbed_sign
                print "adv clip eps: ", perturbed_epsilon
                print "grad: ", gradient
                from scipy.misc import imshow
                imshow(perturbed)
            """


PGD = ProjectedGradientAttack
