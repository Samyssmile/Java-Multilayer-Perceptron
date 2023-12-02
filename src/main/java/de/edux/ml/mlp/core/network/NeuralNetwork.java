package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.network.loader.BatchData;
import de.edux.ml.mlp.core.network.loader.Loader;
import de.edux.ml.mlp.core.network.loader.MetaData;
import de.edux.ml.mlp.core.tensor.Matrix;
import de.edux.ml.mlp.core.transformer.Transform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.LinkedList;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);
    private Engine engine;
    private int epochs;
    private double initialLearningRate;
    private double finalLearningRate = 0.001;
    transient private double learningRate; // TODO rename to ETA ?

    transient private Object lock = new Object();
    private int threads = 8;

    public NeuralNetwork(int batchSize) {
        engine = new Engine(batchSize);
    }

    public NeuralNetwork() {
        this(0);
    }

    public void setLearningRates(double initialLearningRate, double finalLearningRate) {
        this.initialLearningRate = initialLearningRate;
        this.finalLearningRate = finalLearningRate;
    }

    public void setThreads(int threads) {
        this.threads = threads;
    }

    public void add(Transform transform, double... params) {
        engine.add(transform, params);
    }

    public void addLayer(Layer layer) {
        engine.addLayer(layer);
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void fit(Loader trainLoader, Loader evalLoader) {

        learningRate = initialLearningRate;
        for (int epoch = 0; epoch < epochs; epoch++) {
            runEpochLayerBased(trainLoader, true);

            if (evalLoader != null) {
                runEpochLayerBased(evalLoader, false);
            }

            System.out.println("");
            learningRate -= (initialLearningRate - finalLearningRate) / epochs;
        }
    }

    private void runEpochLayerBased(Loader loader, boolean traingMode) {
        loader.open();

        var queue = createBatchTasks(loader, traingMode);
        consumeBatchTasksLayerbased(queue, traingMode);

        loader.close();

/*        if (traingMode) {
            engine.updateWeightsAndBias();
        }*/
    }

    public void setScaleInitialWeights(double scaleInitialWeights) {
        this.engine.setScaleInitialWeights(scaleInitialWeights);
    }


    private synchronized void consumeBatchTasksLayerbased(LinkedList<Future<Matrix>> batches, boolean traingMode) {
        int numberBatches = batches.size();
        int index = 0;

        for (var batch : batches) {
            try {
                var batchResult = batch.get();

            } catch (Exception e) {
                e.printStackTrace();
            }

            int printDot = numberBatches / 30;
            if (traingMode && index++ % printDot == 0) {
                System.out.print(".");
            }
        }

    }

    private LinkedList<Future<Matrix>> createBatchTasks(Loader loader, boolean trainingMode) {
        LinkedList<Future<Matrix>> batches = new LinkedList<>();

        MetaData metaData = loader.getMetaData();
        var numberBatches = metaData.getNumberBatches();

        var executor = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < numberBatches; i++) {
            batches.add(executor.submit(() -> runBatch(loader, trainingMode)));
        }

        executor.shutdown();


        return batches;
    }


    private Matrix runBatch(Loader loader, boolean trainingMode) {
        MetaData metaData = loader.getMetaData();
        BatchData batchData = loader.readBatch();
        int itemsRead = metaData.getItemsRead();
        int inputSize = metaData.getInputSize();
        int expectedSize = metaData.getExpectedSize();

        Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
        Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());

        Matrix batchResult = engine.forwardLayerbased(input);

        if (trainingMode) {
            engine.backwardLayerBased(expected);
        } else {
            engine.evaluateLayerBased(batchResult, expected);
        }

        return batchResult;

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Epochs: %d\n", epochs));
        sb.append(String.format("Initial learning rate: %f, Final learning rate: %f\n", initialLearningRate, finalLearningRate));
        sb.append(String.format("Threads: %d\n", threads));
        sb.append("\nEngine configuration:");
        sb.append("\n----------------------------------------\n");
        sb.append(engine.toString());
        return sb.toString();
    }

    public void setBatchSize(int batchSize) {
        engine.setBatchSize(batchSize);
    }

    public boolean saveModel(String fileName) {
        File file = new File(fileName);
        try (var ds = new ObjectOutputStream(new FileOutputStream(file))) {
            ds.writeObject(this);
            log.info("Model saved to {}", file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

    public static NeuralNetwork load(String fileName) {
        NeuralNetwork model = null;
        File file = new File(fileName);
        if (!file.exists()) {
            return null;
        }
        try (var ds = new ObjectInputStream(new FileInputStream(file))) {
            model = (NeuralNetwork) ds.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        log.info("Model loaded from {}", file.getAbsolutePath());
        return model;
    }

    public double[] predict(Matrix input) {
        return engine.forwardLayerbased(input).getData();
    }

    public LinkedList<Double> getLossHistory() {
        return engine.getLossHistory();
    }

    public LinkedList<Double> getAccuracyHistory() {
        return engine.getAccuracyHistory();
    }
}
