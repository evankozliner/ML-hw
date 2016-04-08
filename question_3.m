k = 300;
n = 2000;
d = 784;
sigma = 1500;
lambda = 0.01;
test_x = load('test79.mat');
train_x = load('train79.mat');
test_x = test_x.d79;
train_x = train_x.d79;
train_x = train_x';
test_x = test_x';
y_train = ones(2000, 1);
y_train(1000:2000, 1) = -1;

w = randn(k, size(train_x, 1))/sigma;
Z = exp(1i*w*train_x);
alpha = (eye(k) * lambda + Z * Z') \ (Z * y_train);
results = alpha'*exp(1i+w*test_x);
labels = sign(real(results))';
numErr = 0;
for i=1:length(labels)
	if y_train(i) ~= labels(i)
		numErr = numErr + 1;
	end
end
numErr/2000
%groups = ones(2000, 1);
%groups(1000:2000, 1) = -1;
%
%weights = randn(d, size(train_x, 2));
%transformed_train = exp( -2 * pi * 1i * (weights * train'));
%alpha = transformed_train \ (transformed_train * groups);

%w = randn(k,d);

%Z = zeros(k,n);
%% Z = zeros(k, size(train_x, 1));
%% R = zeros(n, k);
%
%for i=1:k
%	for j=1:n
%		Z(j,i) = exp(i * w(i,:).' * test_x(j,:));
%	end
%end
%size(Z)
%% kernel = dot(R, R.');
%size(kernel)
%alpha = (Z + lambda*eye(2000,2000)) \ y_train;
%results = test * alpha;
%
%results(results >= 0) = 1;
%results(results < 0) = -1;
%
%loss_7 = sum(results(1:1000,1) == -1)
%loss_9 = sum(results(1000:2000,1) == 1)
%
%(loss_7 + loss_9)/2000
