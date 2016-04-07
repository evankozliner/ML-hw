test = load('test79.mat');
train = load('train79.mat');
test = test.d79;
train = train.d79;

y_train = ones(2000, 1);
y_train(1000:2000, 1) = -1;

% SVM
disp svm;
svm = svmtrain(train, y_train);
results = svmclassify(svm, test);

loss_7 = sum(results(1:1000,1) == -1)
loss_9 = sum(results(1000:2000,1) == 1)

(loss_7 + loss_9)/2000

% Least squares
disp least_squares;
m = train \ y_train;

results = (test * m);

results(results >= 0) = 1;
results(results < 0) = -1;


loss_7 = sum(results(1:1000,1) == -1)
loss_9 = sum(results(1000:2000,1) == 1)

(loss_7 + loss_9)/2000

