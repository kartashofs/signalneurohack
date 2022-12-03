import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle

MODEL_PATH = 'model.onnx'
PKL_PATH = 'pairs.pkl'
OUTPUT_PATH = 'BabyKagglers.npy'

handler = insightface.model_zoo.get_model(MODEL_PATH, providers=['CUDAExecutionProvider'])
handler.prepare(ctx_id=0)


infile = open(PKL_PATH,'rb')

pairs = pickle.load(infile)

infile.close()

probs = []
for pair in pairs:

    input_img = pair[0]

    target_img = pair[1]

    input_img = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB)

    feat_1 = handler.get_feat(input_img)
    target_img= cv2.cvtColor(target_img,cv2.COLOR_GRAY2RGB)

    feat_2 = handler.get_feat(target_img)
    ps = handler.compute_sim(feat_1, feat_2) 

    probs.append(ps)

probs = np.array(probs)
pros = []
for prob in probs:
    if prob > 0.32:
        pro = 1
    else:
        pro = 0
    pros.append(pro)

pros = np.array(pros).reshape(1, len(pros))
pros.save(OUTPUT_PATH)