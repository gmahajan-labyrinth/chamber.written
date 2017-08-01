# What is Nearest Neighbor Methods?

For the [regression problem]({{site.baseurl}}/what/regression-problem/), nearest-neighbor methods use those observations in 
the training set \\(T\\) closest in input space to \\(x\\)
to form \\(\hat{Y}\\) . Specifically, the \\(k\\)-nearest neighbor fit
for \\(\hat{Y}\\) is defined as follows:
\\[\hat{Y}(x)= \dfrac{1}{k} \sum_{x_i\in N_k(x)} y_i\\]

where \\(N_k(x)\\) is the neighborhood of \\(x\\) defined by the 
\\(k\\) closest points \\(x_i\\) in the training sample.