import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import json
import h5py

att = h5py.File('att_vgg.h5')
obj = json.load(open('raw_train.json','r'))
test = []
labels = []
split = 50000 #first sample number of test split
no_test_samples = 10000
for i in range(no_test_samples):
  label = np.zeros([233]).astype('int')	
  test.append(att['maps_with_vgg'][split+i])
  label[obj[split+i]['object_id']] = 1
  labels.append(label)

class _classifier(nn.Module):
    def __init__(self, featureDim , nTargets):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
        	nn.Linear(featureDim,1024),
        	nn.ReLU(),
            nn.Linear(featureDim, 256),
            nn.ReLU(),
            nn.Linear(256, nTargets),
           # nn.Linear(featureDim,nTargets)
        )

    def forward(self, input):
        return self.main(input)

feature_dim = 512
no_classes = len(labels[0])

classifier_test = _classifier(feature_dim,no_classes)
classifier_test = torch.load('trained2_5000.t7')

correct = 0
for j in range(no_test_samples):
 print obj[split+j]['object_id']
 x = test[j]
 #print obj[split+j]
 input = Variable(torch.FloatTensor(x)).view(1,-1)
 predicted = classifier_test(input)
 out = torch.sigmoid(predicted).data > 0.2
 #print out
 maxim,idx = torch.max(torch.sigmoid(predicted).data,1)
 print 'Prediction is:',idx[0]
 if idx[0]==obj[split+j]['object_id']:
 	correct += 1

print 'Accuracy is:',correct
