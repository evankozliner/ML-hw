lambda = 0.01; % most sucessful lamda from problem 2
sigma = 1200; % most sucessful sigma from problem 2
k = 300;
train = load('train79.mat');
test = load('test79.mat');
x = train.d79'; 
x_test = test.d79';
y = ones(2000, 1); % sevens
y(1000:2000, 1) = -1; % nines

% Train the RFF
w = randn(k, size(x, 1)) / sigma;
z = exp(1i * w * x);
alpha = (eye(k) * lambda + z * z') \ (z * y);

% Test 
results = alpha' * exp(1i * w * x_test);
labels = sign(real(results))';
numErr = 0;
loss_7 = sum(labels(1:1000,1) == -1);
loss_9 = sum(labels(1000:2000,1) == 1);
fprintf('RNN gets %.2f %% error\n', (loss_7+loss_9)/20)
