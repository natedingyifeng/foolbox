from __future__ import division
import numpy as np
from .base import Attack


class MomentumIterativeAttack(Attack):

    def clip_ball(self, eta, eps):
        reduc_inc = list(range(1, len(eta.shape)))
        beta = 1e-12
        eta = np.clip(eta, -eps, eps)
        return eta

    def _apply(self, a, u=1.0, epsilon=.3, eps_iter=0.06, nb_iter=10):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()
        momentum = 0
        perturbed = image

        for i in range(nb_iter):
            # compute grad wrt image and label
            grad = a.gradient()
            # normalize gradiet and add
            red_ind = list(range(1, len(grad.shape)))
            # avoid zero grad
            mean_grad = np.mean(np.abs(grad))
            grad = grad / np.maximum(1e-12, mean_grad)
            # add decay factor, default 1.0
            momentum = u * momentum + grad
            normalized_grad = np.sign(momentum)
            scaled_grad = eps_iter * normalized_grad
            perturbed = perturbed + scaled_grad
            perturbed = image + self.clip_ball(perturbed-image, epsilon)
            if min_ is not None and max_ is not None:
                perturbed = np.clip(perturbed, min_, max_)

            logits, is_adversarial = a.predictions(perturbed)
            if is_adversarial:
                print "mim success norm: ", np.max(np.abs(perturbed - image))
                return
            
        return


MIM = MomentumIterativeAttack
