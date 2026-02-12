import jax
import jax.numpy as jnp
import equinox as eqx
from paramax import non_trainable, unwrap

from eqxlearn.base import Regressor
from eqxlearn.gaussian_process.kernels import Kernel

class GaussianProcessRegressor(Regressor):
    """
    Standard GP Regressor.
    Observation noise should be defined inside the kernel (e.g. kernel + WhiteKernel).
    """
    kernel: Kernel
    X: jnp.ndarray
    y: jnp.ndarray
    
    # Jitter is strictly for numerical stability (not trained)
    jitter: float = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        kernel: Kernel, 
        jitter: float = 1e-10
    ):
        """
        Args:
            X: (N, D) Input data
            y: (N,) Target data (1D)
            kernel: Kernel instance
            jitter: Small float added to diagonal for Cholesky stability.
        """
        N, _D = X.shape
        if len(y.shape) != 1 and y.shape[0] != N:
            raise Exception("Incompatible shapes for X and y passed")

        # We use non_trainable to prevent X, y from being updated by the optimizer
        self.X = non_trainable(X)
        self.y = non_trainable(y)
        self.kernel = kernel
        self.jitter = jitter

    def __call__(self, x: jnp.ndarray, return_var: bool = False):
        """
        Predicts mean and variance for a single test point x.
        """
        # Unwrap data to access arrays
        X, y = unwrap(self.X), unwrap(self.y)

        N = X.shape[0]
        k_fn = lambda x1, x2: self.kernel(x1, x2)

        # 1. Compute Kernel Matrix (includes WhiteNoise if present in kernel)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        # 2. Add Jitter (Stability only)
        K_y = K + self.jitter * jnp.eye(N)
        L = jnp.linalg.cholesky(K_y)

        # 3. Compute Cross-Covariance (k_star)
        # Note: If using WhiteKernel, it correctly returns 0 here for x != X
        k_star = jax.vmap(lambda x_train: k_fn(x, x_train))(X)

        # 4. Calculate Mean
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        mu = jnp.dot(k_star, alpha)

        if not return_var:
            return mu

        # 5. Calculate Variance
        # Note: If using WhiteKernel, this includes observation noise variance 
        # (predictive posterior), effectively p(y* | y), not p(f* | y).
        k_star_star = k_fn(x, x)
        v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)
        var = k_star_star - jnp.dot(v, v)
        
        # Clip variance for stability
        return mu, jnp.maximum(var, 1e-12)
    
    def loss(self):
        """Computes Negative Log Marginal Likelihood (NLML)."""
        X, y = unwrap(self.X), unwrap(self.y)
        N = X.shape[0]
        
        # 1. Compute Kernel Matrix
        k_fn = lambda x1, x2: self.kernel(x1, x2)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        # 2. Add Jitter
        K_y = K + self.jitter * jnp.eye(N)
        
        # 3. Cholesky & Solve
        L = jnp.linalg.cholesky(K_y)
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        # 4. Compute NLML
        data_fit = 0.5 * jnp.dot(y, alpha)
        complexity = jnp.sum(jnp.log(jnp.diag(L)))
        constant = 0.5 * N * jnp.log(2 * jnp.pi)
        
        return data_fit + complexity + constant
    
    def plot(self, X, X_train=None, y_train=None, in_axis=None, out_axis=None, uncertainty=False, fig=None, axes=None, clear=True):
        import numpy as np
        import matplotlib.pyplot as plt
        
        if X_train is None and y_train is None:
            X_train = unwrap(self.X)
            y_train = unwrap(self.y)
        
        # 1. Normalize Input
        X = jnp.atleast_2d(jnp.asarray(X)).reshape(X.shape[0], -1)
        input_dim = X.shape[1]
        
        # 2. Run Model
        if uncertainty:
            Y_mean, Y_var = jax.vmap(lambda x: self(x, return_var=True))(X)
        else:
            Y_mean = jax.vmap(lambda x: self(x))(X)
            Y_var = None
            
        if Y_mean.ndim == 1:
            Y_mean = Y_mean[:, None]
            if Y_var is not None:
                Y_var = Y_var[:, None]
        
        # Also ensure training data is 2D for consistent plotting later
        if y_train is not None and y_train.ndim == 1:
             y_train = y_train[:, None]            
            
        # 3. Determine Plotting Mode (1D Line vs 2D Surface)
        is_surface = (in_axis is None) and (input_dim == 2)
        
        # If not a surface and in_axis is still None, default to index 0 for 1D plots
        if in_axis is None and not is_surface:
            in_axis = 0
            
        if out_axis is None:
            out_axis = list(range(Y_mean.shape[1]))

        # --- GRID CALCULATION FIX START ---
        n_plots = len(out_axis)
        # Calculate cols first as ceil(sqrt(N)), then rows as ceil(N / cols)
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        # --- GRID CALCULATION FIX END ---
        
        subplot_kw = {'projection': '3d'} if is_surface else {}

        # 4. Setup Figure and Axes
        if fig is None:
            fig, axes = plt.subplots(rows, cols, subplot_kw=subplot_kw)
        else:
            if clear:
                fig.clf()
                axes = fig.subplots(rows, cols, subplot_kw=subplot_kw)
            elif axes is None:
                axes = fig.subplots(rows, cols, subplot_kw=subplot_kw)

        # Ensure axes is always a flat array
        if not isinstance(axes, (np.ndarray, list, jnp.ndarray)):
            axes = np.array([axes])
        axes = axes.flatten()

        # 5. Plotting Loop
        for i, out_axis_i in enumerate(out_axis):
            ax = axes[i]
            
            in_label = "ALL (2D)" if is_surface else str(in_axis)
            ax.set_title(f'Latent: in={in_label} -> out={out_axis_i}')
            
            _plot_with_variance(X, Y_mean, Y_var, in_axis=in_axis, out_axis=out_axis_i, ax=ax, label='Mean', is_surface=is_surface)
            
            if X_train is not None and y_train is not None:
                if is_surface:
                    ax.scatter(X_train[:,0], X_train[:,1], y_train[:,out_axis_i], color='black', marker='x', label='Train')
                else:
                    ax.scatter(X_train[:,in_axis], y_train[:,out_axis_i], color='black', marker='x', label='Train')
        
            if i == 0:
                ax.legend()
                
        # Hide empty subplots if grid is larger than N plots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # 6. Final Polish
        fig.set_size_inches((cols*5, rows*4))
        fig.tight_layout()
        
        if plt.get_backend() != 'agg':
            plt.pause(0.1) 
        
        return fig, axes
        
def _plot_with_variance(X, Y_mean, Y_var=None, in_axis=None, out_axis=None, ax=None, is_surface=False, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt    
    import logging
    ax = ax or plt.gca()
    
    # --- Data Prep ---
    if in_axis is not None:
        X_plot = X[:, in_axis] # 1D case
    else:
        X_plot = X # 2D case (keep all cols)
        
    Y_mean_plot = Y_mean[:, out_axis]
    
    X_plot = np.asarray(X_plot)
    Y_mean_plot = np.asarray(Y_mean_plot)
    
    Y_std = None
    if Y_var is not None:
        Y_var_plot = np.asarray(Y_var[:, out_axis])
        Y_std = np.sqrt(Y_var_plot)

    # --- Plotting Logic ---
    
    if is_surface:
        # === 3D SURFACE PLOT ===
        # We use plot_trisurf so it works even if X isn't a perfect meshgrid
        
        # Plot Mean (Solid)
        try:
            ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot, cmap='viridis', alpha=0.8, linewidth=0.2, edgecolors='none')
        except:
            logging.warning("Could not plot surface mean")
        
        # Plot Uncertainty (Transparent Shells)
        if Y_std is not None:
            try:
                # Upper bound (Mean + 2 Std)
                ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot + 2*Y_std, color='gray', alpha=0.15, edgecolor='none')
                # Lower bound (Mean - 2 Std)
                ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot - 2*Y_std, color='gray', alpha=0.15, edgecolor='none')
            except:
                logging.warning("Could not plot standard deviation")
            
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        
    else:
        # === 1D LINE PLOT ===
        # Sort X for clean line plotting
        sort_idx = np.argsort(X_plot)
        X_plot = X_plot[sort_idx]
        Y_mean_plot = Y_mean_plot[sort_idx]
        if Y_std is not None:
            Y_std = Y_std[sort_idx]

        # Mean
        ax.plot(X_plot, Y_mean_plot, **kwargs)

        # Variance bands
        if Y_std is not None:
            ax.fill_between(X_plot, Y_mean_plot - 3*Y_std, Y_mean_plot + 3*Y_std, alpha=0.15, color=kwargs.get('color'))
            ax.fill_between(X_plot, Y_mean_plot - 2*Y_std, Y_mean_plot + 2*Y_std, alpha=0.25, color=kwargs.get('color'))
            ax.fill_between(X_plot, Y_mean_plot - 1*Y_std, Y_mean_plot + 1*Y_std, alpha=0.35, color=kwargs.get('color'))

        ax.set_xlabel("x")
        ax.set_ylabel("y")    