import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import LBFGSAttack
from foolbox.attacks import FGSM
from foolbox.attacks import MomentumIterativeAttack as MIM
from foolbox.attacks import ProjectedGradientAttack as PGD
from foolbox.criteria import TargetClassProbability
from foolbox.criteria import Misclassification
import numpy as np
import keras
from keras.datasets import mnist
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from scipy.misc import imsave, imshow, imread
import matplotlib.pyplot as plt
import glob
from train_mnist import madry_mnist_model

keras.backend.set_learning_phase(0)
model = madry_mnist_model()
model.load_weights('mnist_madry.h5')
fmodel = KerasModel(model, bounds=(0, 1))
success = 0.
(images, labels), _ = mnist.load_data()

for image in images[:100]:

    image = image.astype(np.float32)#[:, :, np.newaxis]/255.
    image = image[np.newaxis, :, :, np.newaxis]
    image /= 255.

    test = image.copy()
    preds = model.predict(test)
    label = np.argmax(preds)
    print("Label: ", label)
    #imshow(image[0, :, :, 0])
    
    # run the attack
    print "running the attack"
    print np.max(image[0]), np.min(image[0])
    attack = PGD(model=fmodel, criterion=Misclassification())
    adversarial = attack(image[0], label)

    if adversarial is None:
        print "Did not find an adversarial"
        continue

    # show results
    adversarial_x = adversarial[np.newaxis, :, :, :]
    preds = model.predict(adversarial_x.copy())
    adv_label = np.argmax(preds)
    if adv_label != label:
        success += 1
        print "new label: ", adv_label
    #print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))
    
    diff = (adversarial_x[0] - image)

    # normalize to 0-1 for viewing with matplotlib
    a = adversarial_x[0].copy()
    nx = (a-np.min(a))/(np.max(a)-np.min(a))
    nd = (diff-np.min(diff))/(np.max(diff)-np.min(diff))
    
    max_norm = np.max(np.abs(diff))
    print "max norm: ", max_norm
    print "l2 norm: ", np.linalg.norm(diff)
    print "---------------------------------"
    
    #plt.ion()
    image = image[0, :, :, 0]
    nx = nx[:, :, 0]
    nd = nd[0, :, :, 0]

    plt.subplot(1, 3, 1)
    plt.title("Real")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("eps: {}".format(max_norm))
    plt.imshow(nx, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(nd, cmap='gray')
    plt.savefig('adv_test', format='png')
    plt.show()

print "Success percentage: {}%".format(success)
