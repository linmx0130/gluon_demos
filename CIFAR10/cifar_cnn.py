import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import os
ctx = mx.gpu(0)
batch_size = 64

def transform_builder(flip=False):
    """ 
    Transform function builder
    During training, we need to do some data argumentation like flipping.
    """
    def transform(data, label):
        data = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
        label = label.astype(np.float32)
        if flip:
            if np.random.uniform() < 0.5:
                data = nd.flip(data, axis=2)
        return data, label
    return transform

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform_builder(flip=True)), batch_size=batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform_builder()), batch_size=batch_size, shuffle=False)

class BottleneckBlock(gluon.nn.Block):
    """
    Simple bottlenect block.
    """
    def __init__(self, middle_channels, channels_output, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.conv0 = gluon.nn.Conv2D(channels = middle_channels, kernel_size=(1,1))
        self.bn0 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
        self.relu0 = gluon.nn.Activation(activation='relu')
        self.conv1 = gluon.nn.Conv2D(channels = middle_channels, kernel_size=(3,3), padding=(1,1))
        self.bn1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
        self.relu1 = gluon.nn.Activation(activation='relu')
        self.conv2 = gluon.nn.Conv2D(channels= channels_output, kernel_size=(1,1))
        self.bn2 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)

    def forward(self, x):
        f = self.relu0(self.bn0(self.conv0(x)))
        f = self.relu1(self.bn1(self.conv1(f)))
        f = self.bn2(self.conv2(f))
        return f + x

# build network
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, padding=(1, 1)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D())

    net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))

    net.add(BottleneckBlock(middle_channels=64, channels_output=128))
    
    net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1, 1), strides=(2, 2)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    
    net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, padding=(1, 1)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))

    net.add(BottleneckBlock(middle_channels=128, channels_output=256))

    net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, padding=(1, 1), strides=(2, 2)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    
    net.add(gluon.nn.Conv2D(channels=512, kernel_size=3, padding=(1, 1)))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(BottleneckBlock(middle_channels=256, channels_output=512))

    net.add(gluon.nn.GlobalAvgPool2D())
    net.add(gluon.nn.Dense(10, activation=None))

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
# Momentum SGD
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd':0.0001, 'momentum':0.9})

def eval_acc(net, data_iterator):
    num = 0.0
    denum = 0.0
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        num += nd.sum(predictions == label)
        denum += data.shape[0]
    return (num / denum).asscalar()

smoothing_constant = .01
save_path = "checkpoints/epoch-{}.gluonmodel"
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

for e in range(10):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        curr_loss = nd.mean(loss).asscalar()
        if i==0:
            moving_loss = curr_loss
        else:
            moving_loss = (1-smoothing_constant) * moving_loss + smoothing_constant * curr_loss
    print("Epoch {} loss={}".format(e, moving_loss))
    net.save_params(save_path.format(e))
    print("  Save checkpoint to {}".format(save_path.format(e)))
    test_acc = eval_acc(net, test_data)
    train_acc = eval_acc(net, train_data)
    print("   Test acc = {}, Train acc = {}".format(test_acc, train_acc))

