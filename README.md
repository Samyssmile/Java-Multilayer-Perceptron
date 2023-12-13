# Java Multi Layer Perceptron
Full implementation of a MLP with backpropagation and gradient descent. The MLP is built for my Java ML Library (EDUX).
Check the library out here: [EDUX](https://github.com/Samyssmile/edux)

## Usage
For MNIST training, run MNISTShowcase.java to see the MLP in action. The MLP is able to classify handwritten digits with an accuracy of ~95% after 10 epochs.

## Example

```java
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

## Limitations
Actually its runs only on 1 Thread. We are working on a multithreaded version. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.