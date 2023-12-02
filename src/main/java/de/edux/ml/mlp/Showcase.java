package de.edux.ml.mlp;

import de.edux.ml.mlp.core.network.NeuralNetwork;
import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.ImageLoader;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;

import java.io.File;

public class Showcase {
    public static void main(String[] args) {
        String trainImages = "mnist"+ File.separator+"train-images.idx3-ubyte";
        String trainLabels = "mnist"+ File.separator+"train-labels.idx1-ubyte";
        String testImages = "mnist"+ File.separator+"t10k-images.idx3-ubyte";
        String testLabels = "mnist"+ File.separator+"t10k-labels.idx1-ubyte";

        int batchSize = 100;
        Loader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
        Loader testLoader = new ImageLoader(testImages, testLabels, batchSize);

        MetaData trainMetaData = trainLoader.open();
        int inputSize = trainMetaData.getInputSize();
        int outputSize = trainMetaData.getExpectedSize();
        trainLoader.close();

        NeuralNetwork nn = new NeuralNetwork(batchSize);
        nn.addLayer(new DenseLayer(inputSize, 100));
        nn.addLayer(new ReLuLayer());
        nn.addLayer(new DenseLayer(100, 200));
        nn.addLayer(new ReLuLayer());
        nn.addLayer(new DenseLayer(200, outputSize));
        nn.addLayer(new SoftmaxLayer());

        int threads =1;
        int epochs = 100;

        nn.setThreads(threads);
        nn.setEpochs(epochs);
        nn.setLearningRates(0.05, 0.001);
        nn.setScaleInitialWeights(0.2);

        System.out.println(nn);
        nn.fit(trainLoader, testLoader);
        nn.saveModel("mnist-model.edux");
    }
}
