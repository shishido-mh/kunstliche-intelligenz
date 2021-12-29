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
% Let J be:
% J = J_cost + J_regularization
% Where J_cost is the cost function term and J_regularization is the 
% regularization from the cost function
% Step 1: forward propagation

a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1'; % a_1 (m x n+1) * Theta1'(n+1, n_labels)
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

% Step 2: Transform y from vector to matrix
% Create identity matrix, using eye() function

I = eye(num_labels);

% Create y matrix, to calculate J(theta)

y_matrix = zeros(m, num_labels);

% For each row of y, get a number i (from 1-10). Then, call the i-th row of I,
% which is exactly the vector necessary to perform the calculus for J(theta)
% for each example. Now, paste this vector from I to the i-th row of y_matrix.
% Repeating this procedure m times, the transformation of y(m rows, 1 coluns) to
% y_matrix(m rows, num_labels columns is complete as we need to procede)

for i = 1:m
  y_matrix(i, :)= I(y(i), :);
end

% y_matrix is order (m, num_labels). So is a_3. So, as it is necessary to sum
% all values, an element-wise multiplication calculates all the products
% correctly. Then, just sum all resultant elements.

J_aux = - 1/m .* (y_matrix .* log(a_3) + (1-y_matrix) .* log(1-a_3));

% Transform J_aux from matrix to vector, so it is possible to sum all elements.
% Finally, J can be calculated, as follows:

J_cost = sum(J_aux(:));

% The regularization term is easier: First, square all Theta elements using 
% '.^2'. Then, sum all rows, but remembering to exlude the first column, since
% the first one is always the bias, and it should not be considered following
% the convention. At this moment, the result is a vector with 1 row and 
% num_labels columns, since all rows were grouped by the sum operator. Again, it 
% is necessary to sum all these values, calling the sum() function once more,
% but now using the index 2, to indicate the sum all elements in a column.

J_regularization = lambda/(2 .* m) .* (sum(sum(Theta1(:, 2:end).^2),2) + ...
sum(sum(Theta2(:, 2:end).^2),2));
J = J_cost + J_regularization;
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


for i = 1:m
  
  a_1 = [1, X(i,:)]';
  z_2 = Theta1 * a_1;
  a_2 = sigmoid(z_2);
  a_2 = [1; a_2];
  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);
  y_example = (y_matrix(i,:))';
  delta_3 = a_3 - y_example;
  delta_2 = (Theta2'(2:end,:) * delta_3) .* [sigmoidGradient(z_2)];

  Theta1_grad = Theta1_grad + (delta_2 * a_1');
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  
endfor;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
