package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;

import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

public class DenseLayer implements Layer {
    AtomicReference<Matrix> weights;
    AtomicReference<Matrix> bias;

    ConcurrentLinkedQueue<Matrix> accumulatedWeightGradients = new ConcurrentLinkedQueue<>();
    ConcurrentLinkedQueue<Matrix> accumulatedBiasGradients = new ConcurrentLinkedQueue<>();

    private ReentrantLock lock = new ReentrantLock();


    private final float learningRate = 0.02f; //TODO make configurable

    private final Random random = new Random();

    private Matrix lastInput;

    public DenseLayer(int inputSize, int outputSize) {
        weights = new AtomicReference<>(new Matrix(outputSize, inputSize));
        bias = new AtomicReference<>(new Matrix(outputSize, 1));
        initialize();
    }

    private void initialize() {
        double standartDeviation = Math.sqrt(2.0 / (weights.get().getRows() + weights.get().getCols()));

        for (int i = 0; i < weights.get().getRows(); i++) {
            for (int j = 0; j < weights.get().getCols(); j++) {
                weights.get().set(i, j, random.nextGaussian() * standartDeviation);
            }
        }
        //bias initialization random
        for (int i = 0; i < bias.get().getRows(); i++) {
            for (int j = 0; j < bias.get().getCols(); j++) {
                bias.get().set(i, j, 0);
            }
        }

    }

    @Override
    public Matrix forwardLayerbased(Matrix input) {
        this.lastInput = input;
        return this.weights.get().multiply(input).add(this.bias.get());
    }

    @Override
    public synchronized void updateWeightsAndBias() {
        for (int i = 0; i < accumulatedWeightGradients.size(); i++) {
            Matrix weightGradient = accumulatedWeightGradients.poll();
            Matrix biasGradient = accumulatedBiasGradients.poll();
            double rate = learningRate / lastInput.getCols();

            // Update weights and bias
            this.weights.set(weights.get().modify((index, value) -> value - rate * weightGradient.get(index)));
            this.bias.set(bias.get().modify((row, col, value) -> value - learningRate * biasGradient.get(row)));
        }
        accumulatedWeightGradients.clear();
        accumulatedBiasGradients.clear();

        if (accumulatedWeightGradients.size() > 0) {
            System.out.println(accumulatedWeightGradients.size());
        }
        if (accumulatedBiasGradients.size() > 0) {
            System.out.println(accumulatedBiasGradients.size());
        }

    }


    @Override
    public Matrix backwardLayerBased(Matrix error) {
        Matrix output = weights.get().transpose().multiply(error);
        // Calculate gradient of weights
        Matrix weightsGradient = error.multiply(lastInput.transpose());
        // Calculate gradient of bias
        Matrix biasGradient = error.averageColumn();

        float rate = learningRate / lastInput.getCols();

        // Update weights and bias
        this.weights.set(weights.get().modify((index, value) -> value - rate * weightsGradient.get(index)));
        this.bias.set(bias.get().modify((row, col, value) -> value - learningRate * biasGradient.get(row)));

/*
        synchronized (this) {
            accumulatedWeightGradients.add(weightsGradient);
            accumulatedBiasGradients.add(biasGradient);
        }
*/

        return output;
    }


    @Override
    public String toString() {
        return "DenseLayer " + weights.get().getRows() + "x" + weights.get().getCols();
    }
}
