<strong>ACE_model_UNet.pt</strong> - run with UNet_3d.py, SGD lr=0.07 momentum=0.9, loss =~0.25

Confusion Matrix<br>
[[9.46282178e-01 5.53002060e-04 2.21297690e-04]<br>
[4.17402302e-02 9.83021881e-01 3.99884925e-01]<br>
 [1.19775915e-02 1.64251174e-02 5.99893777e-01]]

<strong>ACE_model_UNet2.pt</strong> - run with UNet_3d.py, Adam lr=0.001 weight_decay=0.0002, loss=0.15, 800 iters

Confusion Matrix<br>
[[7.96585779e-01 7.96832940e-07 0.00000000e+00]<br>
 [7.85427090e-02 9.70717664e-01 9.36974418e-02]<br>
 [1.24871512e-01 2.92815388e-02 9.06302558e-01]]

^^ UNet models needs more training, interested to see with 10,000 iters

___

ARCHIVE

ACE_model.pt - run with net.py, softmax layer at end, loss = ~0.4

ACE_model2.pt - run with net2.py, ReLU layer at end, loss = ~0.8

^^ adapted models probably are not deep enough to learn labels
