function main()
	%cross_validate([.0001, .001, .01, .1], 1:100:10000)
	%cross_validate([.0001, .1], 1:100:200)
	cross_validate([.0001], 1:100:101)
end

function cross_validate(lambdas, sigmas)
	error_rates = zeros(length(sigmas), length(sigmas));
	for i =1:length(lambdas)
		for j =1:length(sigmas)
			error_rates(j,i) = get_err(lambdas(i), sigmas(j));
		end
	end
	[~, n] = min(error_rates(:));
	[l,s] = ind2sub(size(error_rates), n);
	lambdas(l)
	sigmas(s)
	error_rates(s,l)	
end

function percent_err = get_err(lambda, variance)
	test = load('test79.mat');
	train = load('train79.mat');
	test = test.d79;
	train = train.d79;
	
	y_train = ones(2000, 1);
	y_train(1000:2000, 1) = -1;
	kernel = get_kernel(2000, variance, train, train);

	alpha = (kernel + lambda*eye(2000,2000)) \ y_train;

	test_kernel = get_kernel(2000, variance, train, test);
	%result = ( test_kernel + lambda * eye(2000,2000) ) * alpha;
	y = 0;
	labels = zeros(2000,1);

	for j=1:2000
		y = 0;
		for i=1:2000
			y = y + alpha(i,:) * test_kernel(i,j);
		end
		labels(j,:) = sign(y);
	end
	
	%err = ((1000 - sum(labels(1:1000))) + (1000 + sum(labels(1:1000))))/20
	num_err = 0;
	for i=1:2000
		if labels(i,:) ~= y_train(i, :)
			num_err = num_err + 1;
		end
	end
	percent_err = num_err/2000
end

function approx_kernel = rff()
	
end

function kernel = get_kernel(dimensions, variance, vectors, vectors2)
	kernal = zeros(dimensions, dimensions);
	for i = 1:dimensions
		for j = 1:dimensions
			kernel(i,j) = gaussian(vectors(i,:),vectors2(j,:),variance);
		end
	end
end
function point = gaussian(x, y, sigma)
	point = exp((-1)*(norm(x - y))^2/sigma^2);
end
