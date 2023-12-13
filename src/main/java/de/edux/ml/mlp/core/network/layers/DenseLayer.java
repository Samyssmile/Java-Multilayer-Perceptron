package de.edux.ml.mlp.core.network.layers;
import de.edux.ml.mlp.core.network.Layer;
import de.edux.ml.mlp.core.tensor.Matrix;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class DenseLayer implements Layer {
  AtomicReference<Matrix> weights;
  AtomicReference<Matrix> bias;

  private float learningRate = 0.01f; // TODO make configurable

  private final ThreadLocalRandom random = ThreadLocalRandom.current();
  private Matrix weightGradients;
  private Matrix biasGradients;
  private Matrix lastInput;

  private AtomicInteger gradientAccumulations = new AtomicInteger(0);


  public DenseLayer(int inputSize, int outputSize) {
    weights = new AtomicReference<>(new Matrix(outputSize, inputSize));
    bias = new AtomicReference<>(new Matrix(outputSize, 1));
    weightGradients = new Matrix(outputSize, inputSize);
    biasGradients = new Matrix(outputSize, 1);
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
    if (gradientAccumulations.get() > 0) {
      float rate = learningRate/this.gradientAccumulations.get();
      this.weights.set(this.weights.get().subtract(weightGradients.multiply(rate)));
      this.bias.set(this.bias.get().subtract(biasGradients.multiply(rate )));

      weightGradients.fill(0);
      biasGradients.fill(0);
      gradientAccumulations.set(0);
    }
  }

  @Override
  public Matrix backwardLayerBased(Matrix error) {
    gradientAccumulations.incrementAndGet();
    Matrix output = weights.get().transpose().multiply(error);
    // Akkumulieren der Gradienten
    weightGradients = weightGradients.add(error.multiply(lastInput.transpose()));
    biasGradients = biasGradients.add(error.averageColumn());

    return output;
  }

  @Override
  public String toString() {
    return "DenseLayer " + weights.get().getRows() + "x" + weights.get().getCols();
  }
}