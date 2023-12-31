package de.edux.ml.mlp;

import de.edux.ml.mlp.core.network.NetworkBuilder;
import de.edux.ml.mlp.core.network.NeuralNetwork;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.ImageLoader;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;

import java.io.File;

public class MNISTShowcase {
  public static void main(String[] args) {
    String trainImages = "mnist" + File.separator + "train-images.idx3-ubyte";
    String trainLabels = "mnist" + File.separator + "train-labels.idx1-ubyte";
    String testImages = "mnist" + File.separator + "t10k-images.idx3-ubyte";
    String testLabels = "mnist" + File.separator + "t10k-labels.idx1-ubyte";

    int batchSize = 100;
    int threads = 1;
    int epochs = 20;
    float initialLearningRate = 0.1f;
    float finalLearningRate = 0.001f;

    Loader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
    Loader testLoader = new ImageLoader(testImages, testLabels, batchSize);

    MetaData trainMetaData = trainLoader.open();
    int inputSize = trainMetaData.getInputSize();
    int outputSize = trainMetaData.getExpectedSize();
    trainLoader.close();

    // Training from scratch
    new NetworkBuilder()
        .addLayer(new DenseLayer(inputSize, 64))
        .addLayer(new ReLuLayer())
        .addLayer(new DenseLayer(64, outputSize))
        .addLayer(new SoftmaxLayer())
        .withBatchSize(batchSize)
        .withLearningRates(initialLearningRate, finalLearningRate)
        .withThreads(threads)
        .withEpochs(epochs)
        .build()
        .fit(trainLoader, testLoader)
        .saveModel("model.edux"); //Save the model

    // Loading a model and continue training
    NeuralNetwork nn = new NetworkBuilder().withEpochs(10).loadModel("model.edux").fit(trainLoader, testLoader);
  }
}
