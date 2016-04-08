k = 392;
n = 2000;
d = 784;
lambda = 0.01;
test_x = load('test79.mat');
train_x = load('train79.mat');
test_x = test_x.d79;
train_x = train_x.d79;
y_train = ones(2000, 1);
y_train(1000:2000, 1) = -1;

w = randn(k,d);
Z = zeros(k,n);
% Z = zeros(k, size(train_x, 1));
% R = zeros(n, k);

for i=1:k
	for j=1:n
		Z(j,i) = exp(i * w(i,:).' * test_x(j,:));
	end
end
size(Z)
% kernel = dot(R, R.');
size(kernel)
alpha = (Z + lambda*eye(2000,2000)) \ y_train;
results = test * alpha;

results(results >= 0) = 1;
results(results < 0) = -1;

loss_7 = sum(results(1:1000,1) == -1)
loss_9 = sum(results(1000:2000,1) == 1)

(loss_7 + loss_9)/2000
