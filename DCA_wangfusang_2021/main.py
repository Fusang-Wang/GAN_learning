# 2021.1.4 wang fusang email:1290391034@qq.com

import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np
import DCA

############################ data ##############################
(train_images, train_labels), (test_images, test_quitlabels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
sample_train = train_images[0:500]
sample_label = train_labels[0:500]
print(sample_train.shape)
print(sample_label.shape)

########################## network ###########################

cross_entropy =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def compute_loss(model,images,labels):
    output = model(images)
    #print(output.shape)
    #print(labels.shape)
    loss = cross_entropy(labels,output)
    return loss

model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.summary()

################# training #################### 
X = sample_train
y = sample_label
epochs = range(21)
losses =[]
with tf.GradientTape() as tape:
        Loss = compute_loss(model,X,y)
        losses.append(Loss)
        print(Loss)
        gradients = tape.gradient(Loss,model.trainable_variables)

optimizer =DCA.DCA(model,gradients,Loss)
print(optimizer.rho)

for i in range(20):
    with tf.GradientTape() as tape:
        Loss = compute_loss(model,X,y)
        losses.append(Loss)
        print(Loss)
        gradients = tape.gradient(Loss,model.trainable_variables)
        approx_l = optimizer.approx_loss()
        loss_xk_1 = compute_loss(model,X,y)
        #print("loss(xk_1):",loss_xk_1)
        #print("approx-loss:",approx_l)
        optimizer.apply_gradient(approx_l,loss_xk_1)

##################loss函数很小的事停止#####################
#####################SGD 收敛差距和variance有关，learning rate，################
#################SGD 收敛技巧 收敛半径##################
 ####################加速boosted#########################



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, 'o', label='train')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)