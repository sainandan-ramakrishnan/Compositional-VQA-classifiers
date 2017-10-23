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
                #nn.Linear(featureDim,1024),
        	    #nn.ReLU(),
                #nn.Linear(1024,512),
                #nn.ReLU(),
                nn.Linear(featureDim, nTargets),
                nn.LogSoftmax()
        
        )

    def forward(self, input):
        return self.main(input)

def get_batches():
    batch_start = 0
    batch_end = batch_size
    random.shuffle(sample_idx)
    all_train_batches = []
    all_label_batches = []
    count = 0
    while (batch_end <= no_training_samples - crop): 
    	idx = sample_idx[batch_start:batch_end]
    	this_train_batch = []
    	this_label_batch = []
    	this_train_batch = [train[0][id] for id in idx]
    	this_label_batch = [labels[0][id] for id in idx]
    	all_train_batches.append(this_train_batch)
    	all_label_batches.append(this_label_batch)
    	if batch_end == no_training_samples:
          break
    	batch_start += batch_size
    	batch_end = min(batch_end + batch_size, no_training_samples)
    	count += 1
    if len(all_train_batches) == len(all_label_batches):
    	print 'Batch sampling complete'
    else:
    	print 'Batch sizes unequal'
    return (all_train_batches,all_label_batches)


features = h5py.File('compo_features.h5')
answers = h5py.File('labels.h5')
train = []
labels = []
no_training_samples = features['features'].shape[0]
print no_training_samples 
batch_size = 100
crop = 158 
train.append(features['features'][0:no_training_samples - crop])     #train[0] is the data array
labels.append(answers['labels'][0:no_training_samples - crop])       #labels[0] is the label array
no_classes = 1000 
feature_dim = 780
vqa = answering_net(feature_dim,no_classes).cuda()
print vqa  
sample_idx = range(no_training_samples - crop)
optimizer = optim.Adam(vqa.parameters(), weight_decay = 2e-5)
criterion = nn.NLLLoss()
epochs = 150
(T,L) = get_batches()

#print torch.FloatTensor(torch.from_numpy(np.array(T[0]))).view(10,-1)
#print torch.from_numpy(np.array(L[0])) 
for epoch in range(epochs):
	losses = []
    
	for j in range(len(T)):
    	 labelsv = Variable(torch.from_numpy(np.array(L[j])).cuda())
    	 inputv = Variable(torch.FloatTensor(torch.from_numpy(np.array(T[j]))).cuda()).view(batch_size, -1)
    	 output_ans = vqa(inputv)
    	 loss = criterion(output_ans, labelsv)
    	 optimizer.zero_grad()
    	 loss.backward()
    	 optimizer.step()
    	 losses.append(loss.data.mean())
	print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
	if (epoch+1)%10 == 0:
		 torch.save(vqa.state_dict(),'models/final_mlp_' + str(epoch+1) + '.t7')

