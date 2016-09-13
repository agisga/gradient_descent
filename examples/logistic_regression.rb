require 'nmatrix'
require 'nmatrix/lapacke'
require 'daru'
require 'statsample'
require 'statsample-glm'
require 'gradient_descent'

#--- Data generation

n = 1000

df = Daru::DataFrame.new({
  a: Statsample::Shorthand.rnorm(n),
  b: Statsample::Shorthand.rnorm(n),
  c: Statsample::Shorthand.rnorm(n),
  d: Statsample::Shorthand.rnorm(n),
})
x = df.to_nmatrix

beta = NMatrix.new([4,1], [3, 5, -2, 2], dtype: :float64)
p = df.a * beta[0,0] + df.b * beta[1,0] + df.c * beta[2,0] + df.d * beta[3,0] 
p.map! { |i| 1/(1+Math::exp(-i)) }
y = p.map do |prob|
  unif_draw = rand
  unif_draw < prob ? 1 : 0
end
df[:y] = Daru::Vector.new(y)
y = NMatrix.new([n,1], y)

#--- logisitic regression objective function and its gradient

f_logistic = Proc.new do |beta|
  n = y.shape[0]

  summands = (0...n).map do |i|
    xi_beta = (x.row(i).dot beta).to_f
    (-y[i, 0] * xi_beta) + Math::log(1.0 + Math::exp(xi_beta))
  end

  summands.inject(:+)
end

gradf_logistic = Proc.new do |beta|
  n = y.shape[0]

  p = NMatrix.new([n,1], -999.999, dtype: :float64)
  index = 0
  x.each_row do |r|
    rbeta = (r.dot(beta)).to_f
    p[index] = 1.0 / (1.0 + Math::exp(-rbeta))
    index += 1
  end

  x.transpose.dot(p - y)
end

#--- determine optimal step size for gradient descent
#FIXME: this can be done more efficiently

u, s, vt = (x.transpose.dot x).gesvd
lipschitz = 0.25 * s.max.to_f
step_size = 1.0 / lipschitz

#--- find the optimal parameters with the gradient descent algorithm

puts "------------------------"
puts "(1) Gradient descent without backtracking:"
puts "Step size: #{step_size}"

opt = GradientDescent.optimize(x0: NMatrix.new([4,1], 0, 
                                               dtype: :float64),
                               t: step_size, backtrack: FALSE, 
                               f: f_logistic, tol: 1e-6,
                               max_iter: 10000,
                               gradf: gradf_logistic)
puts "Number of iterations: #{opt[:iterations]}"
puts "Estimated optimum:"
puts opt[:solution].to_a

puts "------------------------"
puts "(2) Gradient descent with backtracking:"

opt_backtrack = GradientDescent.optimize(x0: NMatrix.new([4,1], 0, 
                                                     dtype: :float64),
                                     t: step_size, backtrack: TRUE, 
                                     f: f_logistic, tol: 1e-6,
                                     max_iter: 10000,
                                     gradf: gradf_logistic)
puts "Number of iterations: #{opt_backtrack[:iterations]}"
puts "Estimated optimum:"
puts opt_backtrack[:solution].to_a

#--- find the optimal parameters with statsample-glm

puts "------------------------"
puts "(3) Statsample-glm algorithm:"

opt2 = Statsample::GLM.compute df, :y, :logistic, algorithm: :mle,
  epsilon: 1e-6
puts "Number of iterations: #{opt2.iterations}"
puts "Estimated optimum:"
puts opt2.coefficients.to_a

#--- benchmarking the different methods
# The benchmarks crash if statsample-glm was applied to the data frame once already... 
# So, for the below benchmarks to work the "(3) Statsample-glm..." stuff must be commented out.

require 'benchmark'
Benchmark.bm do |x|

  x.report("statsample-glm") do
    Statsample::GLM.compute df, :y, :logistic, algorithm: :mle, epsilon: 1e-6
  end

  x.report("GD without backtracking") do
    GradientDescent.optimize(x0: NMatrix.new([4,1], 0, dtype: :float64),
                             t: step_size, backtrack: FALSE, 
                             f: f_logistic, tol: 1e-6,
                             max_iter: 10000,
                             gradf: gradf_logistic)
  end

  x.report("GD with backtracking") do
    GradientDescent.optimize(x0: NMatrix.new([4,1], 0, 
                                         dtype: :float64),
                         t: step_size, backtrack: TRUE, 
                         f: f_logistic, tol: 1e-6,
                         max_iter: 10000,
                         gradf: gradf_logistic)
  end
end
