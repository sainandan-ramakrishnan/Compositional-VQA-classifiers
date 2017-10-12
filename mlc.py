import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import json
import h5py

att = h5py.File('attended_vgg.h5')
obj = json.load(open('train_50.json','r'))
train = []
labels = []
for i in range(50):
  label = np.zeros([233]).astype('int')
  train.append(att['maps_with_vgg'][i])
  label[obj[i]['object_id']] = 1
  labels.append(label)

class _classifier(nn.Module):
    def __init__(self, featureDim , nTargets):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(featureDim, nTargets),
        )

    def forward(self, input):
        return self.main(input)

no_classes = len(labels[0]) # 3 target classes, for us it is equal to no. of objects.
feature_dim = 512
classifier = _classifier(feature_dim,no_classes)

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()

epochs = 5
for epoch in range(epochs):
    losses = []
    for i, sample in enumerate(train):
        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        
        output = classifier(inputv)
        loss = criterion(output, labelsv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))

trained_model = torch.save(classifier,'trained1.t7')

classifier_test = _classifier(feature_dim,no_classes)
classifier_test = torch.load('trained1.t7')
x = train[1]
print x
input = Variable(torch.FloatTensor(x)).view(1,-1)
predicted = classifier_test(input)
out = torch.sigmoid(predicted).data > 0.5
print out 


