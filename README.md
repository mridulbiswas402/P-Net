# P-Net: A CNN Model with Pseudo Dense Layers
This is the official implementation of "A CNN Model with Pseudo Dense Layers:
Some Case Studies on Medical Image
Classificatio" (ISBI, 2024)

### Overall workflow:
![architecture](https://github.com/mridulbiswas402/P-Net/blob/main/results/PNet.png?raw=true)

##  Visualization on MNIST:
![Mnist](https://github.com/mridulbiswas402/P-Net/blob/main/results/pnet_mnist.png?raw=true)

##  Visualization on cats&dog:
![cats&gods](https://github.com/mridulbiswas402/P-Net/blob/main/results/pnet_cats.png?raw=true)


## How to use
-Fork the repository.<br/>
-Download the LC25000 and Brain Tumour MRI dataset datasets.<br/>
-Run the jupyter notebook in demo folder to train the model and generate results.<br/> 
-OR you can can go to respective dataset folder (brain,colon-lung) and run any of respective model file
(densenet.py,resnet.py,efficientnet.py)<br/>
-Make sure to change the paths according to your requirement.<br/>

## Results
### Brain tumor MRI dataset

| Model                 | Params  | Accuracy | Precision |  Recall  |
|-----------------------|---------|----------|-----------|----------|
| ResNet+P-Net          |  2.8M   |  0.984   |  0.985    |  0.985   |
| DenseNet+P-Net        |  1.2M   |  0.992   |  0.992    |  0.992   |
| EfficientNet-b0+P-Net |  3.5M   |  0.992   |  0.993    |  0.992   |


### Colon cancer histopathological datase

| Model                 | Params  | Accuracy | Precision |  Recall  |
|-----------------------|---------|----------|-----------|----------|
| ResNet+P-Net          |  2.8M   |  0.999   |  0.999    |  0.999   |
| DenseNet+P-Net        |  1.2M   |  0.999   |  0.999    |  0.999   |
| EfficientNet-b0+P-Net |  3.5M   |  1.000   |  1.000    |  1.000   |

### Lung cancer histopathological dataset.

| Model                 | Params  | Accuracy | Precision |  Recall  |
|-----------------------|---------|----------|-----------|----------|
| ResNet+P-Net          |  2.8M   |  0.994   |  0.994    |  0.994   |
| DenseNet+P-Net        |  1.2M   |  1.000   |  1.000    |  1.000   |
| EfficientNet-b0+P-Net |  3.5M   |  1.000   |  1.000    |  1.000   |


## Authors :
Mridul Biswas<br/>
Ritodeep Sikdar<br/> 
Ram Sarkar<br/>
Mahantapas Kundu<br/>

## Citation :
Please do cite our paper in case you find it useful for your research.<br/>
Citation-<br/>

<br/>
-Link to our paper-<br/>
