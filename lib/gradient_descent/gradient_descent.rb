# Copyright (c) 2016 Alexej Gossmann 
 
# TODO: docs
class GradientDescent 

  attr_reader :optimum

  # TODO: docs
  #
  # === Arguments
  #
  # * +x0+        - initial parameter estimate
  # * +t+         - step size
  # * +max_iter+  - maximum number of iterations
  # * +tol+       - convergence tolerance
  # * +backtrack+ - boolean for whether backtracking line search should be used or not
  # * +f+         - a +Proc+ object that returns the value of the
  #   objective function at any given parameter
  # * +gradf+     - a +Proc+ object that returns the value of the gradient of the
  #   objective function at any given parameter
  #
  def initialize(x0: , t: , max_iter: 1e5, tol: 1e-3, backtrack: TRUE, &f, &gradf)
    if backtrack then
      @optimum = gradient_descent_backtrack(x0: x0, t: t, max_iter: max_iter, tol: tol, &f, &gradf)
    else
      @optimum = gradient_descent_fixed(x0: x0, t: t, max_iter: max_iter, tol: tol, &f, &gradf)
    end
  end

  # TODO: docs
  #
  # === Arguments
  #
  # * +x+         - current parameter estimate
  # * +t+         - step size
  # * +gradf+     - a +Proc+ object that returns the value of the gradient of the
  #   objective function at any given parameter
  #
  def gradient_step(x: , t: , &gradf)
    x - t * gradf.call(x)
  end

  # TODO: docs
  #
  # === Arguments
  #
  # * +x0+        - initial parameter estimate
  # * +t+         - step size
  # * +max_iter+  - maximum number of iterations
  # * +tol+       - convergence tolerance
  # * +f+         - a +Proc+ object that returns the value of the
  #   objective function at any given parameter
  # * +gradf+     - a +Proc+ object that returns the value of the gradient of the
  #   objective function at any given parameter
  #
  def gradient_descent_fixed(x0: , t: , max_iter: 1e5, tol: 1e-3, &f, &gradf)
    x_old   = x0
    f_old   = f.call(x_old)
    iter    = 0
    optimal = FALSE

    max_iter.times do
      x_new = gradient_step(x: x_old, t: t, &gradf)
      f_new = f.call(x_new)
      iter += 1

      relative_error = (f_new - f_old).abs / (f_old.abs + 1.0)
      if relative_error < tol then
        optimal = TRUE
        break
      end

      x_old = x_new
      f_old = f_new
    end

    return {:solution => x_new, :value => f_new, :optimal => optimal,
            :iterations => iter, :relative_error => relative_error}
  end

  # TODO: docs
  #
  # === Arguments
  #
  # * +x+          - current parameter estimate
  # * +gradf_at_x+ - value of the gradient of the objective function
  #   evaluated at +x+
  # * +t0+         - initial step size
  # * +alpha+      - the backtracking parameter
  # * +beta+       - the decrementing multiplier
  # * +f+          - a +Proc+ object that returns the value of the
  #   objective function at any given parameter
  #
  def backtrack(x: , gradf_at_x: , t0:, alpha: 0.5, beta: 0.9, &f)
    s = t0

    while f.call(x - s * gradf_at_x) > (f.call(x) - alpha * s * gradf_at_x.norm2)
      s = beta * s
    end

    return s
  end

  # TODO: docs
  #
  # === Arguments
  #
  # * +x0+        - initial parameter estimate
  # * +t+         - step size
  # * +max_iter+  - maximum number of iterations
  # * +tol+       - convergence tolerance
  # * +f+         - a +Proc+ object that returns the value of the
  #   objective function at any given parameter
  # * +gradf+     - a +Proc+ object that returns the value of the gradient of the
  #   objective function at any given parameter
  #
  def gradient_descent_backtrack(x0: , t: , max_iter: 1e5, tol: 1e-3, &f, &gradf)
    x_old   = x0
    f_old   = f.call(x_old)
    iter    = 0
    optimal = FALSE
    t0      = t

    max_iter.times do
      t = backtrack(x: x_old, gradf_at_x: gradf.call(x_old), t0: t0, &f)

      x_new = gradient_step(x: x_old, t: t, &gradf)
      f_new = f.call(x_new)
      iter += 1

      relative_error = (f_new - f_old).abs / (f_old.abs + 1.0)
      if relative_error < tol then
        optimal = TRUE
        break
      end

      x_old = x_new
      f_old = f_new
    end

    return {:solution => x_new, :value => f_new, :optimal => optimal,
            :iterations => iter, :relative_error => relative_error}
  end
end
