# CNN Classifier - AlexNet

### Problem Statement

The task for this assignment is to test your designed neural network models for image classification problem and classify the remotely sensed images using deep learning architectures. There are 7 classes: Basketball court, beach, forest, railway, tennis court, water pool, others

### Data Preprocessing 

The given training set had 560 images. Data Augmentation has been used to optimize the dataset. Augmentor library for python has been used for the task. Final training set has 3080 images. Some augmentation tasks performed are as follows
  
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

p.zoom_random(probability=0.3,percentage_area =0.8, randomise_percentage_area=True)

p.flip_left_right(probability=0.4)

p.flip_top_bottom(probability=0.4)

p.random_erasing(probability=0.2,rectangle_area=0.8)

### Model Architecture

![AlexNet]("..\alexnet.png")

AlexNet Model has been implemented for this task with slight modifications.

Framework: Pytorch

LocationCNN(

  (layer1): Sequential(

    (0): Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

    (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    (2): ReLU()

  )

  (layer2): Sequential(

    (0): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    (3): ReLU()

  )

  (layer3): Sequential(

    (0): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    (3): ReLU()

  )

  (layer4): Sequential(

    (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    (2): ReLU()

  )

  (layer5): Sequential(

    (0): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1))

    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    (3): ReLU()

  )

  (flatten): Flatten()

  (layer6): Sequential(

    (0): Linear(in_features=4608, out_features=98, bias=True)

    (1): Dropout2d(p=0.25, inplace=False)

    (2): Linear(in_features=98, out_features=7, bias=True)

    (3): Softmax(dim=None)

  )

)

### Result

The model achieved 94% Accuracy on training for 20 epochs in an batch size of 77 for 2541 images and validation set consisting of 539 images.
<hr>

![Accuracy]("./accuracy.png)

<hr>

![TrainingLoss]("./trainloss.png")


