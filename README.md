# Java Multilayer Perceptron
Full implementation of a Neural Network in Java from scratch. I built this Multilayer Perceptron  for my Java ML Library (EDUX).
Check the library out here: [EDUX](https://github.com/Samyssmile/edux) and dont forget to star it ;-)

## Usage
For MNIST training, run **MNISTShowcase.java** to see the MLP in action. The MLP is able to classify handwritten digits with an accuracy of ~95% after 10 epochs.

## Example

```
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 32))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(32, outputSize))
        .addLayer(new SoftmaxLayer())
        .withBatchSize(batchSize)
        .withLearningRates(initialLearningRate, finalLearningRate)
        .withThreads(threads)
        .withEpochs(epochs)
        .build()
        .fit(trainLoader, testLoader)
        .saveModel("model.edux");
```

## Output

```output
........................Epoch: 1, Loss: 0,28, Accuracy: 91,68
........................Epoch: 2, Loss: 0,23, Accuracy: 93,22
........................Epoch: 3, Loss: 0,20, Accuracy: 94,41
........................Epoch: 4, Loss: 0,17, Accuracy: 94,97
........................Epoch: 5, Loss: 0,15, Accuracy: 95,42
........................Epoch: 6, Loss: 0,14, Accuracy: 95,80
........................Epoch: 7, Loss: 0,13, Accuracy: 96,07
........................Epoch: 8, Loss: 0,13, Accuracy: 96,27
........................Epoch: 9, Loss: 0,12, Accuracy: 96,43
........................Epoch: 10, Loss: 0,12, Accuracy: 96,58
........................Epoch: 11, Loss: 0,11, Accuracy: 96,78
........................Epoch: 12, Loss: 0,11, Accuracy: 96,84
........................Epoch: 13, Loss: 0,11, Accuracy: 96,87
........................Epoch: 14, Loss: 0,11, Accuracy: 96,88
........................Epoch: 15, Loss: 0,10, Accuracy: 96,93
........................Epoch: 16, Loss: 0,10, Accuracy: 97,00
........................Epoch: 17, Loss: 0,10, Accuracy: 97,05
........................Epoch: 18, Loss: 0,10, Accuracy: 97,06
........................Epoch: 19, Loss: 0,10, Accuracy: 97,07
........................Epoch: 20, Loss: 0,10, Accuracy: 97,04
```
## Features
- [x] Multilayer Perceptron
- [x] ReLu Layer
- [x] Softmax Layer
- [x] Cross Entropy Loss
- [x] MNIST Loader
- [x] Save and Load Models, continue training afrer loading
- [x] Super Fast

## Limitations
Actually its runs only on 1 Thread. We are working on a multithreaded version. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.