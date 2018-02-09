import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import LBFGSAttack
from foolbox.attacks import FGSM
from foolbox.attacks import MomentumIterativeAttack as MIM
from foolbox.criteria import TargetClassProbability
from foolbox.criteria import Misclassification
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from scipy.misc import imsave, imshow, imread
import matplotlib.pyplot as plt
import glob

image_dir = './imgs'

keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)


paths = glob.glob(image_dir+'/*.png')
for path in paths:
    image = imread(path).astype(np.float32)

    test = image.copy().astype(np.float32)
    preds = kmodel.predict(preprocess_input(np.expand_dims(test, 0)))
    label = np.argmax(preds)
    print("Top 3 predictions (regular: ", decode_predictions(preds, top=3))
    
    # run the attack
    print "running the attack"
    attack = MIM(model=fmodel, criterion=Misclassification())
    adversarial = attack(image[:, :, ::-1], label)

    if adversarial is None:
        print "Did not find an adversarial"
        continue
    # show results
    print(np.argmax(fmodel.predictions(adversarial)))
    print(foolbox.utils.softmax(fmodel.predictions(adversarial))[781])
    adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
    preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
    print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))

    diff = (adversarial_rgb[0] - image)
    a = adversarial_rgb[0].copy()
    nx = (a-np.min(a))/(np.max(a)-np.min(a))
    nd = (diff-np.min(diff))/(np.max(diff)-np.min(diff))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(nx)
    plt.subplot(1, 2, 2)
    plt.imshow(nd)
    plt.savefig('adv_test', format='png')
    plt.show()
    print "l2 norm: ", np.linalg.norm(diff)
    print "max norm: ", np.max(np.abs(adversarial_rgb-image))

