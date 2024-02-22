#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

static matrix add_matrices(matrix a, matrix b);
static matrix scale_matrix(double scalar, matrix input);

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
                x = 1.0 / (1.0 + expf(-x));
            } else if (a == RELU){
                // TODO
                x = MAX(x, 0.0);
            } else if (a == LRELU){
                // TODO
                x = MAX(x, x * 0.1);
            } else if (a == SOFTMAX){
                // TODO
                x = expf(x);
            }
            m.data[i][j] = x;
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for(j = 0; j < m.cols; j++){
                double x = m.data[i][j];
                m.data[i][j] = x / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            double gradient = 0.0;
            if (a == LOGISTIC) {
               gradient = x * (1 - x); 
            } else if (a == RELU) {
                gradient = x > 0;
            } else if (a == LRELU) {
                if (x > 0) {
                  gradient = 1.0;
                } else {
                  gradient = 0.1;
                }
            } else if (a == SOFTMAX) {
              gradient = 1.0;
            }
            d.data[i][j] *= gradient;
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    ACTIVATION activationFunction = l->activation;
    matrix weights = l->w;
    matrix out = matrix_mult_matrix(in, weights);
    activate_matrix(out, activationFunction);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix dw = matrix_mult_matrix(transpose_matrix(l -> in), delta);
    l->dw = dw;

    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix dx = matrix_mult_matrix(delta, transpose_matrix(l->w));

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    //         deltaW = part1  - part2 + part3
    matrix part2 = scale_matrix(-1.0 * decay, l->w);
    matrix part3 = scale_matrix(momentum, l->v);
    matrix part1AndPart2 = add_matrices(l->dw, part2);
    matrix delta_w = add_matrices(part1AndPart2, part3);
    free_matrix(part1AndPart2);
    free_matrix(part3);
    free_matrix(part2);

    // save it to l->v
    free_matrix(l -> v);
    l -> v = delta_w;

    // Update l->w
    // weights = weights + learing rate * old
    // weights = weights + rate * old
    // weights = weights + part2
    part2 = scale_matrix(rate, l->v);
    matrix newWeights = add_matrices(l->w, part2);
    free_matrix(l->w);
    free_matrix(part2);
    l->w = newWeights;

    // Remember to free any intermediate results to avoid memory leaks
    // see above for frees
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}

static matrix scale_matrix(double scalar, matrix input) {
    matrix output = copy_matrix(input);
    for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
        output.data[i][j] *= scalar;
      }
    }
    return output;
}

static matrix add_matrices(matrix a, matrix b) {
    matrix output = copy_matrix(a);
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        output.data[i][j] += b.data[i][j];
      }
    }
    return output;
}

// Questions 
//
// 2.1.1 What are the training and test accuracy values you get? Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// TODO
// training accuracy: 90.33 %
// test accuracy:     90.75 %
// The training accuracy tells us how well the model does on the training set, the testing accuracy tells us how the model performs on data it has not seen before
// The training accuracy is important to evaluate how well the training works, the testing accuracy is important since it tells us how the model generalizes to out of sample data
//
// 2.1.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// TODO
// rate    training accuracy     testing accuracy      loss
// 10^1          9.87%                 9.80%           -nan
// 10^0          9.87%                 9.80%           -nan
// 10^-1         91.94%                91.65%          0.24
// 10^-2         90.33%                90.75%          0.29
// 10^-3         85.85%                86.91%          0.59
//
// As learning rate decreases, accuracy increased to a certain point and then it went down again. 
// When our accuracy improved, our loss also decreased (excluding nans) 
// This shows us that choosing a good learning rate is important, and for this model, a rate of 10^-1 would likely be best
//
// 2.1.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// TODO
// Higher values of decay lead to less accurate testing and training accuracy. However, once the decay goes below 10^-2, lowering it further does not seem to improve testing and training accuracy
//
// 2.2.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// TODO
// Trying the other models, RELU had the best training and test accuracy of 92.21% and 92.32% respectively, followed by LRELU, LOGISTIC and then SOFTMAX, with SOFTMAX being by far the worst
//
// 2.2.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// TODO
// The best learning rate is 10^-1, with a training accuracy of 96.13% and test accuracy of 95.64%
//
// 2.2.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// TODO
// Adding a bit of decay made the training and testing accuracy both decrease, which means that it did not help. This happened probably because the model was already quite good, and decay caused some lost progress
//
// 2.2.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// TODO
// The lowest decay (10^-4) had the best accuracy, though it was extremely close to 10^-3. Higher decay causes training accuracy to drop, likely because the model loses some progress that it would have made
//
// 3.1.1 What is the best training accuracy and testing accuracy? Summarize all the hyperparameter combinations you tried.
// TODO
// I tested all combinations of rates = [.1, .01, 0.001, 0.0001] and decays = [.1, .01, 0.001, 0.0001]
// I fixed batch at 128, iters at 3000, and momentum at 0.9
// The best training accuracy was ENTER ACCURACY HERE, which happened with a rate of RATE HERE, and decay of DECAY HERE
// The best testing accuracy was ENTER ACCURACY HERE, which happened with a rate of RATE HERE, and decay of DECAY HERE 
