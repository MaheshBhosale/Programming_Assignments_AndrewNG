function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
x=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
  s=sigmoid(X(i,:) * theta);
  J += ((-y(i)*log(s)) -((1-y(i))*log(1-s)));
 endfor
J = J/m; 
for j=2:n
  
  x = (x + (theta(j)*theta(j)));
  endfor

x = (( x * lambda ) /( 2 * m ));
J = J + x;

s=0;
for i=1:m
      s += (((sigmoid(X(i,:) * theta)) - y(i)) * X(i,1));
  endfor
  
s = s/m;
grad(1) = s;
s = 0;

for j=2:n
  for i=1:m
      s += (((sigmoid(X(i,:) * theta))-y(i))*X(i,j));
  endfor
  s = s / m;
  z = s + (( lambda / m ) * theta( j ));
  grad( j ) = z;
  s = 0;
  z = 0;
 endfor
