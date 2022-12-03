
'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')

# other modules
import os
import numpy as np
from utils.utils_SH import *

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
from model.defineHourglass_512_gray_skip import *
import pickle
import insightface


os.system('unzip DPR-master.zip')
os.system('cd DPR-master/')

MODEL_PATH = 'model.onnx'
PKL_PATH = 'pairs.pkl'
OUTPUT_PATH = 'BabyKagglers.npy'

def do_it(Img):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    #-----------------------------------------------------------------

    modelFolder = '/content/DPR-master/trained_model'

    # load model
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
    # my_network.cuda()
    my_network.train(False)

    lightFolder = '/content/DPR-master/data/example_light'

    row, col = Img.shape
    img = cv2.resize(Img, (512, 512))
    Lab = img
    inputL = Lab
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL))

    resultlabs = []
    for i in range(7):
        sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
        sh = sh[0:9]
        sh = sh * 0.7

        #--------------------------------------------------
        # rendering half-sphere
        sh = np.squeeze(sh)
        shading = get_shading(normal, sh)
        value = np.percentile(shading, 95)
        ind = shading > value
        shading[ind] = value
        shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        shading = (shading *255.0).astype(np.uint8)
        shading = np.reshape(shading, (256, 256))
        shading = shading * valid
        #--------------------------------------------------

        #----------------------------------------------
        #  rendering images using the network
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh))
        outputImg, outputSH  = my_network(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_GRAY2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        resultlabs.append(resultLab)

    return resultlabs


handler = insightface.model_zoo.get_model(MODEL_PATH, providers=['CUDAExecutionProvider'])
handler.prepare(ctx_id=0, providers=['CUDAExecutionProvider'])


infile = open(PKL_PATH,'rb')

pairs = pickle.load(infile)

infile.close()

probs = []
o=0
for ll, pair in enumerate(pairs):
    if ll % 100 == 0:
      print(o)
      o += 100
    input_img = pair[0]

    target_img = pair[1]
    pss = []

    dd_input = do_it(input_img)
    dd_target = do_it(target_img)
    for i, t in zip(dd_input, dd_target):

        input_img = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

        feat_1 = handler.get_feat(np.atleast_3d(input_img))
        target_img= cv2.cvtColor(t,cv2.COLOR_BGR2RGB)

        feat_2 = handler.get_feat(np.atleast_3d(target_img))
        ps = handler.compute_sim(feat_1, feat_2) 
        pss.append(ps)
    probs.append(sum(pss) / len(pss))

probs = np.array(probs)
pros = []
for prob in probs:
    if prob > 0.32:
        pro = 1
    else:
        pro = prob
    pros.append(pro)

prons = np.array(pros).reshape(1, len(pros))
np.save(OUTPUT_PATH, prons)
