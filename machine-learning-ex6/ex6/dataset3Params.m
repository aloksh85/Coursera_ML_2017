function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_test =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_test =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
prediction_errors = zeros(length(C_test),length(sigma_test));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i = 1:length(C_test)
  C = C_test(i);
  for  j = 1:length(sigma_test)
    
    sigma = sigma_test(j);
 %     disp("Combination: "); i,j,C,sigma
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma),1e-3, 20);
    pred =svmPredict(model,Xval);
    prediction_errors(i,j) = mean(double(pred ~= yval));
    endfor
endfor
[min_v1, idx1] = min(prediction_errors,[],2);
[min_v2,idx2] = min(min_v1);
C= C_test(idx2);
sigma = sigma_test(idx1(idx2));

% =========================================================================
endfunction