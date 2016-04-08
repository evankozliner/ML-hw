function [] = hw2_3_test()
train = load('train79.mat');
test = load('test79.mat');
trainX = train.d79;
trainY = [repmat(-1,1000,1);ones(1000,1)];
lambda = 0.01;
sigma = 1500;
k = 300;
X = trainX';
Xtest = test.d79';
Y = trainY;
w = randn(k, size(X, 1)) / sigma;
Z = exp(1i*w*X);
alpha = (eye(k)*lambda+Z*Z')\(Z*Y);
results = alpha'*exp(1i*w*Xtest);
labels = sign(real(results))';
numErr = 0;
for i=1:length(labels)
    if Y(i) ~= labels(i)
        numErr = numErr + 1;
    end
end
fprintf('Random Fourier Features: %.2f %% error\n', (numErr / 2000) * 100);
end
