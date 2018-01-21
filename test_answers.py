import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import json
import h5py
import random

class answering_net(nn.Module):
    def __init__(self, featureDim , nTargets):
        super(answering_net, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(featureDim,1024),
        	nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, nTargets),
                nn.LogSoftmax()
        
        )

    def forward(self, input):
        return self.main(input)

annotations = json.load(open('annotations_val.json','r'))
params = json.load(open('params_1.json','r'))
features_test = h5py.File('compo_features_u_test.h5')
answers_test = h5py.File('labels_test.h5')
ques_test = json.load(open('vqa_raw_test.json','r'))
no_classes = 1000 
feature_dim = 780
vqa = answering_net(feature_dim,no_classes).cuda()
vqa.load_state_dict(torch.load('models_u/best_u_model.t7'))
print vqa 
correct = 0
#no_test_samples = 200000
no_test_samples = features_test['features'].shape[0]
results = []
for j in range(no_test_samples):
    inputv = Variable(torch.FloatTensor(features_test['features'][j]).cuda()).view(1, -1)
    predicted_ans = vqa(inputv)
    max_confidence,ans_idx = torch.max(predicted_ans.data,1)
    print(max_confidence) 
    print "Prediction",j+1,"is:",params['ix_to_ans'][str(ans_idx[0]+1)]
    agree = 0
    for i in range(10):
      if params['ix_to_ans'][str(ans_idx[0]+1)] in annotations['annotations'][j]['answers'][i]['answer']:
        agree += 1
    
    agree = min(agree/3.0,1.0)
    correct += agree
    results.append({'answer':params['ix_to_ans'][str(ans_idx[0]+1)],'question':ques_test[j]['question']})
print "Accuracy is:", correct*100.0/no_test_samples,"% out of",no_test_samples,"samples"
#print answers_test['labels'][0]
weights = list(vqa.parameters())[0].data.cpu().numpy()
#print weights[7].shape[0]
w_comp = weights[7][0:267]
w_san = weights[7][268:]
print sum(abs(w_comp))/float(len(w_comp))
print sum(abs(w_san))/float(len(w_san)) 
json.dump(results,open('our_results_53.2%.json','w'))
