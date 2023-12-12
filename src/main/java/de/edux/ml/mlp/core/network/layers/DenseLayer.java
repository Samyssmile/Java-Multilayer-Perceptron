package de.edux.ml.mlp.core.network.layers;

import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

public class DenseLayer implements Layer {
  AtomicReference<Matrix> weights;
  AtomicReference<Matrix> bias;

  private final float learningRate = 0.05f; // TODO make configurable

  private final Random random = new Random();

  private Matrix lastInput;

  private int gradientContributions = 0;


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
    System.out.println(this.gradientContributions);
    this.gradientContributions=0;
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
    this.weights.set(
        weights.get().modify((index, value) -> value - rate * weightsGradient.get(index)));
    this.bias.set(
        bias.get().modify((row, col, value) -> value - learningRate * biasGradient.get(row)));

    gradientContributions++;
    return output;
  }

  @Override
  public String toString() {
    return "DenseLayer " + weights.get().getRows() + "x" + weights.get().getCols();
  }
}
