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
        perturbed = perturbed + np.random.uniform(-epsilon, epsilon, image.shape)
        perturbed = np.clip(perturbed, min_, max_)
        for _ in range(steps):
            gradient = a.gradient(perturbed)
            perturbed += lr * np.sign(gradient)
            perturbed = np.clip(perturbed, image-epsilon, image+epsilon)
            perturbed = np.clip(perturbed, min_, max_)
            a.predictions(perturbed)


PGD = ProjectedGradientAttack
