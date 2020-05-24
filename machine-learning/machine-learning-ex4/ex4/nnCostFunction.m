function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

inputLayerWithBias = [ones(1, m); X'];
hiddenLayer = sigmoid(Theta1 * inputLayerWithBias);
hiddenLayerWithBias = [ones(1, m); hiddenLayer];
outputLayer = sigmoid(Theta2 * hiddenLayerWithBias);

recodedY = zeros(num_labels, m);
for i = 1 : m
	recodedY(y(i), i) = 1;
end

complexity = (-recodedY .* log(outputLayer)) - ((1 - recodedY) .* log(1 - outputLayer));
sumOverK = sum(complexity);
sumOverMAndK = sum(sumOverK);
unregularizedCost = (1/m) * sumOverMAndK;

theta1ForRegularization = Theta1;
theta1ForRegularization(:, 1) = 0;
sumOverTheta1 = sum(sum(theta1ForRegularization .^ 2, 2));

theta2ForRegularization = Theta2;
theta2ForRegularization(:, 1) = 0;
sumOverTheta2 = sum(sum(theta2ForRegularization .^ 2, 2));

regularization = (lambda / (2 * m)) * (sumOverTheta1 + sumOverTheta2);
J = unregularizedCost + regularization;
% -------------------------------------------------------------

% =========================================================================
z2 = Theta1 * inputLayerWithBias; % node values of the 2nd layer before activation function is applied
z2 = [ones(1, m); z2];

outputErrors = outputLayer - recodedY;
hiddenErrors = (Theta2' * outputErrors) .* sigmoidGradient(z2);

accumulatedDeltas2 = zeros(size(Theta2));
accumulatedDeltas2 = accumulatedDeltas2 + (outputErrors * hiddenLayerWithBias');
Theta2_grad = (1 / m) * accumulatedDeltas2;

theta2ForRegularization = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
regularization = (lambda / m) .* theta2ForRegularization;
Theta2_grad = Theta2_grad + regularization;

accumulatedDeltas1 = zeros(size(Theta1));
accumulatedDeltas1 = accumulatedDeltas1 + (hiddenErrors(2:end, :) * inputLayerWithBias');
Theta1_grad = (1 / m) * accumulatedDeltas1;

theta1ForRegularization = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
regularization = (lambda / m) .* theta1ForRegularization;
Theta1_grad = Theta1_grad + regularization;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
