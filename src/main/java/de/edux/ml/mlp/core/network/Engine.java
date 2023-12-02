package de.edux.ml.mlp.core.network;

import de.edux.ml.mlp.core.network.loss.LossFunction;
import de.edux.ml.mlp.core.network.loss.LossFunctions;
import de.edux.ml.mlp.core.tensor.Matrix;
import de.edux.ml.mlp.core.transformer.Transform;
import de.edux.ml.mlp.exceptions.UnsupportedLayerException;
import de.edux.ml.mlp.exceptions.UnsupportedLossFunction;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Random;

//Network
public class Engine implements Layer, Serializable {
    private static final long serialVersionUID = 1L;
    private LinkedList<Double> lossHistory = new LinkedList<>();
    private LinkedList<Double> accuracyHistory = new LinkedList<>();
    private LinkedList<Transform> transforms = new LinkedList<>();
    private LinkedList<Matrix> weights = new LinkedList<>();
    private LinkedList<Matrix> biases = new LinkedList<>();

    private LinkedList<Layer> layers = new LinkedList<>();

    private LossFunction lossFunction = LossFunction.CROSS_ENTROPY;

    private double scaleInitialWeights = 0.9;

    transient private RunningAverages runningAverages;

    public Engine(int batchSize) {
        this.batchSize = batchSize;
        initAverageMetrics();
    }

    private boolean storeInputError = false;
    private int batchSize;

    public BatchResult forward(Matrix input) {
        BatchResult batchResult = new BatchResult();
        Matrix output = input;
        int denseIndex = 0;

        batchResult.addIo(output);
        for (Transform transform : transforms) {
            switch (transform) {
                case DENSE:
                    batchResult.addWeightInput(output);
                    output = weights.get(denseIndex).multiply(output).add(biases.get(denseIndex));
                    denseIndex++;
                    break;
                case RELU:
                    output = output.relu();
                    break;
                case SOFTMAX:
                    output = output.softmax();
                    break;
            }
            batchResult.addIo(output);
        }
        return batchResult;
    }

    public void setScaleInitialWeights(double scaleInitialWeights) {
        this.scaleInitialWeights = scaleInitialWeights;
    }

    @Override
    public Matrix backwardLayerBased(Matrix error) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            error = layers.get(i).backwardLayerBased(error);
        }

        return error;
    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        Matrix output = input;
        for (Layer layer : layers) {
            output = layer.forwardLayerbased(output);
        }
        return output;

    }


    public synchronized double evaluateLayerBased(Matrix predicted, Matrix expected) {
        if (LossFunction.CROSS_ENTROPY != lossFunction) {
            throw new UnsupportedLossFunction("Only Cross Entropy is supported.");
        }

        double loss = LossFunctions.crossEntropy(expected, predicted).averageColumn().get(0);
        Matrix predictions = predicted.getGreatestRowNumber();
        Matrix actual = expected.getGreatestRowNumber();

        int correct = 0;
        for (int i = 0; i < actual.getCols(); i++) {
            if (predictions.get(i) == actual.get(i)) {
                correct++;
            }
        }

        double percentCorrect = (100.0 * correct) / actual.getCols();
        this.accuracyHistory.add(percentCorrect);
        this.lossHistory.add(loss);
        if (this.runningAverages == null) {
            initAverageMetrics();
        }
        this.runningAverages.add(loss, percentCorrect);
        return loss;
    }

    private void initAverageMetrics() {
        this.runningAverages = new RunningAverages(2, this.batchSize,(callNumber, averages) -> {
            System.out.printf("Epoch: %d, Loss: %.2f, Accuracy: %.2f\n", callNumber, averages[0], averages[1]);
        });
    }

    private final Random random = new Random();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void add(Transform transform, double... params) {
        switch (transform) {
            case DENSE:
                int numberOfNeurons = (int) params[0];
                int weightsPerNeuron = weights.size() > 0 ? weights.getLast().getRows() : (int) params[1];
                Matrix weight = new Matrix(numberOfNeurons, weightsPerNeuron, (index) -> scaleInitialWeights * random.nextGaussian());
                Matrix bias = new Matrix(numberOfNeurons, 1, (index) -> 0);
                weights.add(weight);
                biases.add(bias);
                break;
            case RELU:
                break;
            case SOFTMAX:
                break;
        }

        transforms.add(transform);
    }

    public boolean isStoreInputError() {
        return storeInputError;
    }

    public void setStoreInputError(boolean storeInputError) {
        this.storeInputError = storeInputError;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("\nScale Initial Weights:  %3f\n", scaleInitialWeights));
        sb.append("\nTransforms:\n");

        int weightIndex = 0;
        for (Transform transform : transforms) {
            sb.append(transform);
            sb.append(" ");
            switch (transform) {
                case DENSE:
                    sb.append(weights.get(weightIndex).toString(false));
                    weightIndex++;
                    break;
                case RELU:
                    break;
                case SOFTMAX:
                    break;
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public String toStringLayerBased() {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.toString());
            sb.append("\n");
        }
        return sb.toString();
    }



    public LinkedList<Double> getLossHistory() {
        return lossHistory;
    }

    public LinkedList<Double> getAccuracyHistory() {
        return accuracyHistory;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public void updateWeightsAndBias() {
        for (Layer layer : layers) {
            layer.updateWeightsAndBias();
        }
    }

}
