# Talk 4: Optimisation and gradient descent

## Theory
- **Essential**: Bach 5.1; from start of 5.2 to start of 5.2.1; 5.2.5; from start of 5.4 to start of 5.4.1.

- **Nice to have**: 5.2.1; 5.4.2; 5.4.4

- **Not so important**: everything else in chapter 5.

## Code

We will revisit the least-squares linear regression example from talk 3.

- Load 1D data set as in that example.
- Instead of using the closed-form expression, we will use gradient descent to find parameters. To do this we will use `jax` (seen briefly in talk 1) to differentiate the objective function and compute gradients; can do this by hand ('vanilla' gradient descent / SGD) and possibly also introduce [`optax`](https://optax.readthedocs.io/en/latest/) library which has more sophisticated stochastic gradient descent algorithms.

Learning `jax` and `optax` will be a useful warm-up for the section on neural networks later.
