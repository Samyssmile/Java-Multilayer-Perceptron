package de.edux.ml.mlp.core.network;


import de.edux.ml.mlp.core.network.layers.DenseLayer;
import de.edux.ml.mlp.core.network.layers.ReLuLayer;
import de.edux.ml.mlp.core.network.layers.SoftmaxLayer;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.test.TestLoader;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class FullTrainingTest {

    @Test
    public void test() {
        int threads = Runtime.getRuntime().availableProcessors();
        int epochs = 3;
        var start = System.currentTimeMillis();
        System.out.println("Available processors (cores): " +
                Runtime.getRuntime().availableProcessors());

        NeuralNetwork nn = null;

        if (nn == null) {
            System.out.println("Unable to load model");
            int inputRows = 500;
            int outputRows = 3;

            nn = new NeuralNetwork(100);
            nn.addLayer(new DenseLayer(inputRows, 100));
            nn.addLayer(new ReLuLayer());
            nn.addLayer(new DenseLayer(100, 256));
            nn.addLayer(new ReLuLayer());
            nn.addLayer(new DenseLayer(256, outputRows));
            nn.addLayer(new SoftmaxLayer());

            nn.setThreads(threads);
            nn.setEpochs(epochs);
            nn.setLearningRates(0.1, 0.001);
        }

        System.out.println(nn);
        Loader trainLoader = new TestLoader(60_000, 100);
        Loader evalLoader = new TestLoader(10_000, 100);

        nn.fit(trainLoader, evalLoader);
        /*      nn.saveModel("model.edux");*/
        var end = System.currentTimeMillis();
        String result = "On " + threads + " threads it took " + (end - start) + " ms";

        assertTrue(nn.getAccuracyHistory().getLast() > nn.getAccuracyHistory().getFirst());
        assertTrue(nn.getLossHistory().getLast()<nn.getLossHistory().getFirst());
    }
}
