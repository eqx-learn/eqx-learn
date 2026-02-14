import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_surface(
    model, 
    X, 
    X_train=None, 
    y_train=None, 
    in_axis=None, 
    out_axis=None, 
    uncertainty=False, 
    fig=None, 
    axes=None, 
    clear=True
):
    """
    Generic plotting function for regression models (1D lines or 2D surfaces).
    
    Args:
        model: A callable (or object with .predict) taking (x, return_var=bool).
        X: (N, D) Test input locations to create the surface/line.
        X_train: (Optional) (M, D) Training inputs to scatter plot.
        y_train: (Optional) (M, K) Training targets to scatter plot.
        in_axis: Index of the input dimension to plot (for 1D slices). 
                 If None and D=2, plots a 3D surface.
        out_axis: List of output dimensions to plot. Defaults to all outputs.
        uncertainty: Whether to query the model for variance.
    """

    # 1. Normalize Input
    # Ensure X is 2D: (N_samples, N_features)
    X = jnp.atleast_2d(jnp.asarray(X))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    N_samples, input_dim = X.shape
    
    # 2. Run Model Prediction
    # We define a helper to call the model, handling both __call__ and .predict
    def predict_fn(x):
        if hasattr(model, 'predict'):
            return model.predict(x, return_var=uncertainty)
        else:
            return model(x, return_var=uncertainty)

    if uncertainty:
        # Vmap over the batch dimension of X
        Y_mean, Y_var = jax.vmap(predict_fn)(X)
    else:
        Y_mean = jax.vmap(predict_fn)(X)
        Y_var = None

    # Ensure Y is at least 2D: (N_samples, N_outputs)
    if Y_mean.ndim == 1:
        Y_mean = Y_mean[:, None]
        if Y_var is not None:
            Y_var = Y_var[:, None]
            
    # Normalize Training Data if provided
    if X_train is not None:
        X_train = np.asarray(X_train)
    if y_train is not None:
        y_train = np.asarray(y_train)
        if y_train.ndim == 1:
            y_train = y_train[:, None]

    # 3. Determine Plotting Mode
    # If input is 2D and no specific axis is requested, defaults to Surface plot
    is_surface = (in_axis is None) and (input_dim == 2)
    
    # If not surface and no axis specified, default to 0 for 1D plot
    if in_axis is None and not is_surface:
        in_axis = 0

    # Default to plotting all output dimensions
    if out_axis is None:
        out_axis = list(range(Y_mean.shape[1]))
    if not isinstance(out_axis, (list, tuple, np.ndarray)):
        out_axis = [out_axis]

    # 4. Grid Calculation
    n_plots = len(out_axis)
    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))
    
    subplot_kw = {'projection': '3d'} if is_surface else {}

    # 5. Setup Figure and Axes
    if fig is None:
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        # Create subplots manually to handle potential mix of 2d/3d correctly if needed,
        # but here we assume uniform type based on input_dim.
        axes_list = []
        for i in range(n_plots):
            axes_list.append(fig.add_subplot(rows, cols, i+1, **subplot_kw))
        axes = np.array(axes_list)
    else:
        if clear:
            fig.clf()
        if axes is None:
            axes = fig.subplots(rows, cols, subplot_kw=subplot_kw)
        
    # Ensure axes is iterable
    if not isinstance(axes, (np.ndarray, list)):
        axes = np.array([axes])
    axes = axes.flatten()

    # 6. Plotting Loop
    for i, out_idx in enumerate(out_axis):
        if i >= len(axes): 
            break
            
        ax = axes[i]
        in_label = "ALL (2D)" if is_surface else f"Dim {in_axis}"
        ax.set_title(f'Latent: {in_label} -> Out {out_idx}')

        # Call internal helper to draw the data
        _draw_on_axis(
            X, Y_mean, Y_var, 
            in_axis=in_axis, 
            out_axis=out_idx, 
            ax=ax, 
            is_surface=is_surface
        )

        # Scatter Training Data
        if X_train is not None and y_train is not None:
            if is_surface:
                # For 3D: Plot x1, x2, y
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], y_train[:, out_idx], 
                    color='black', marker='x', label='Train', alpha=0.6
                )
            else:
                # For 1D: Plot x[in_axis], y
                ax.scatter(
                    X_train[:, in_axis], y_train[:, out_idx], 
                    color='black', marker='x', label='Train', zorder=10
                )
        
        if i == 0:
            ax.legend()

    # Hide unused axes
    for j in range(len(out_axis), len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    return fig, axes

def _draw_on_axis(X, Y_mean, Y_var, in_axis, out_axis, ax, is_surface):
    """Helper to handle the specific matplotlib calls for 1D lines or 3D surfaces."""
    
    # Extract specific output dimension
    y_mu = np.asarray(Y_mean[:, out_axis])
    y_std = np.sqrt(np.asarray(Y_var[:, out_axis])) if Y_var is not None else None

    if is_surface:
        # === 3D Surface Plot ===
        x1 = np.asarray(X[:, 0])
        x2 = np.asarray(X[:, 1])

        # Plot Mean Surface
        try:
            ax.plot_trisurf(x1, x2, y_mu, cmap='viridis', alpha=0.8, linewidth=0.2, edgecolors='none')
        except Exception as e:
            print(f"Warning: Failed to plot trisurf: {e}")

        # Plot Uncertainty Shells (if requested)
        if y_std is not None:
            try:
                # Upper bound (Mean + 2 Std)
                ax.plot_trisurf(x1, x2, y_mu + 2 * y_std, color='gray', alpha=0.15, edgecolor='none')
                # Lower bound (Mean - 2 Std)
                ax.plot_trisurf(x1, x2, y_mu - 2 * y_std, color='gray', alpha=0.15, edgecolor='none')
            except:
                pass

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("y")

    else:
        # === 1D Line Plot ===
        x_flat = np.asarray(X[:, in_axis])
        
        # Sort for clean line rendering
        sort_idx = np.argsort(x_flat)
        x_sorted = x_flat[sort_idx]
        y_mu_sorted = y_mu[sort_idx]
        
        ax.plot(x_sorted, y_mu_sorted, label='Mean', color='C0')

        if y_std is not None:
            y_std_sorted = y_std[sort_idx]
            # 3 Sigma
            ax.fill_between(x_sorted, y_mu_sorted - 3*y_std_sorted, y_mu_sorted + 3*y_std_sorted, alpha=0.1, color='C0')
            # 2 Sigma
            ax.fill_between(x_sorted, y_mu_sorted - 2*y_std_sorted, y_mu_sorted + 2*y_std_sorted, alpha=0.2, color='C0')
            # 1 Sigma
            ax.fill_between(x_sorted, y_mu_sorted - 1*y_std_sorted, y_mu_sorted + 1*y_std_sorted, alpha=0.3, color='C0')

        ax.set_xlabel(f"$x_{in_axis}$")
        ax.set_ylabel("y")