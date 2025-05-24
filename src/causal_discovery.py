import numpy as np
import pandas as pd
from dodiscover import make_context
from dodiscover.toporder.score import SCORE
from sklearn.preprocessing import StandardScaler
import networkx as nx
from lassonet import LassoNetRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import time


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

def find_sparsest_lasso_alpha(X, y, test_size=0.2, random_state=42, alphas=None, cv=5, max_iter=10000):
    """
    Find the sparsest alpha for a Lasso model within one standard error of the best MSE.
    
    Parameters:
    - X: Feature matrix (pandas DataFrame or numpy array)
    - y: Target vector (pandas Series or numpy array)
    - test_size: Proportion of data for test set (default: 0.2)
    - random_state: Random seed for reproducibility (default: 42)
    - alphas: List of alpha values to test (default: logspace from 10^-4 to 10^1)
    - cv: Number of cross-validation folds (default: 5)
    - max_iter: Maximum iterations for Lasso convergence (default: 10000)
    
    Returns:
    - dict: Contains sparsest_alpha, best_alpha, final_model, mse_final, r2_final, selected_features,
            best_mse, best_non_zero, sparsest_non_zero
    """
    # Default alpha range if none provided
    if alphas is None:
        alphas = np.logspace(-2, 10, 30)  # From 0.0001 to 10

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit LassoCV to find best alpha
    lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=random_state, max_iter=max_iter)
    lasso_cv.fit(X_train, y_train)

    # Get MSE and non-zero coefficient counts for each alpha
    mse_path = lasso_cv.mse_path_.mean(axis=1)  # Mean MSE across CV folds
    non_zero_counts = []
    for alpha in lasso_cv.alphas_:
        lasso_temp = Lasso(alpha=alpha, max_iter=max_iter)
        lasso_temp.fit(X_train, y_train)
        non_zero_counts.append(np.sum(lasso_temp.coef_ != 0))

    # Best alpha (lowest MSE)
    best_alpha = lasso_cv.alpha_
    print(f'best alpha: {best_alpha}, ')
    print(non_zero_counts)
    best_mse = mse_path[np.where(lasso_cv.alphas_ == best_alpha)[0][0]]
    best_non_zero = non_zero_counts[np.where(lasso_cv.alphas_ == best_alpha)[0][0]]

    # Find sparsest alpha within one standard error
    mse_se = lasso_cv.mse_path_.std(axis=1) / np.sqrt(lasso_cv.mse_path_.shape[1])  # Standard error
    mse_threshold = best_mse + mse_se[np.where(lasso_cv.alphas_ == best_alpha)[0][0]]
    valid_alphas = [(alpha, mse, nz) for alpha, mse, nz in zip(lasso_cv.alphas_, mse_path, non_zero_counts) if mse <= mse_threshold]
    sparsest_alpha = max(valid_alphas, key=lambda x: x[0])[0]  # Highest alpha within threshold
    sparsest_non_zero = min([nz for _, _, nz in valid_alphas])  # Minimum non-zero coefficients

    # Train final model with sparsest alpha
    final_model = Lasso(alpha=sparsest_alpha, max_iter=max_iter)
    final_model.fit(X_train, y_train)

    # Evaluate final model
    y_pred = final_model.predict(X_test)
    mse_final = mean_squared_error(y_test, y_pred)
    r2_final = r2_score(y_test, y_pred)

    # Get selected features
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
    coef = pd.Series(final_model.coef_, index=feature_names)
    selected_features = coef[coef != 0]

    return {
        'sparsest_alpha': sparsest_alpha,
        'best_alpha': best_alpha,
        'final_model': final_model,
        'mse_final': mse_final,
        'r2_final': r2_final,
        'selected_features': selected_features,
        'best_mse': best_mse,
        'best_non_zero': best_non_zero,
        'sparsest_non_zero': sparsest_non_zero
    }

def display_lasso_results(results):
    """
    Display the results of the Lasso sparsity analysis.
    
    Parameters:
    - results: Dictionary returned by find_sparsest_lasso_alpha
    """
    print(f"Best Alpha (lowest MSE): {results['best_alpha']:.6f}")
    print(f"Best MSE: {results['best_mse']:.6f}")
    print(f"Non-zero coefficients (best alpha): {results['best_non_zero']}")
    print(f"\nSparsest Alpha (within 1 SE): {results['sparsest_alpha']:.6f}")
    print(f"Non-zero coefficients (sparsest alpha): {results['sparsest_non_zero']}")
    print(f"\nFinal Model Performance (sparsest alpha):")
    print(f"Mean Squared Error: {results['mse_final']:.6f}")
    print(f"R-squared: {results['r2_final']:.6f}")
    print("\nSelected Features:")
    print(results['selected_features'])

def generate_scm(n_samples, relations, noise_std=1.0, seed=None):
    """
    Generate data from a nonlinear additive SCM with Gaussian noise.
    
    Parameters:
        n_samples (int): Number of samples to generate.
        relations (list of callables): Each function defines X_i in terms of previous Xs and noise.
        noise_std (float or list): Standard deviation(s) for Gaussian noise.
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Generated data with columns X1, X2, ..., Xn.
    """
    rng = np.random.default_rng(seed)
    n_vars = len(relations)
    X = np.zeros((n_samples, n_vars))
    if isinstance(noise_std, (float, int)):
        noise_std = [noise_std] * n_vars

    for i, rel in enumerate(relations):
        noise = rng.normal(0, noise_std[i], size=n_samples)
        X[:, i] = rel(X, noise)
    columns = [f"X{i+1}" for i in range(n_vars)]
    
    return pd.DataFrame(X, columns=columns)

def get_topological_order(df_scaled, method='SCORE', **kwargs):
    """
    Get the topological order of variables using specified causal discovery method.
    
    Parameters:
        df_scaled (pd.DataFrame): Scaled input data
        method (str): Causal discovery method to use ('SCORE', 'DAS', 'NoGAM', 'CAM')
        **kwargs: Additional arguments to pass to the causal discovery method
        
    Returns:
        tuple: (topological_order, causal_graph)
            topological_order: List of variable names in topological order
            causal_graph: Learned causal graph
    """
    # Create context for causal discovery
    context = make_context().variables(data=df_scaled).build()
    
    # Initialize the causal discovery method
    if method == 'SCORE':
        cd_method = SCORE(**kwargs)
    elif method == 'DAS':
        from dodiscover.toporder.das import DAS
        cd_method = DAS(**kwargs)
    elif method == 'NoGAM':
        from dodiscover.toporder.nogam import NoGAM
        cd_method = NoGAM(**kwargs)
    elif method == 'CAM':
        from dodiscover.toporder.cam import CAM
        cd_method = CAM(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    cd_method.learn_graph(df_scaled, context)
    
    graph = cd_method.graph_
    order_graph = cd_method.order_graph_
    
    # Get topological order directly as variable names
    topological_order = list(nx.topological_sort(order_graph))
    
    return topological_order, graph

def predict_functional_form(df_scaled, target_var, parent_vars):
    """
    Predict the functional form of the last variable in the topological order
    using LassoNet.
    
    Parameters:
        df_scaled (pd.DataFrame): Scaled input data
        target_var (str): Target variable
        parent_vars (list): List of parent variables
        
    Returns:
        tuple: (model, feature_importance)
            model: Fitted LassoNet model
            feature_importance: Dictionary of feature importance scores
    """
    start_time = time.time()
    # Prepare data
    X = df_scaled[parent_vars].values
    y = df_scaled[target_var].values
    
    lasso_results = find_sparsest_lasso_alpha(X, y)
    display_lasso_results(lasso_results)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit LassoNet with optimized parameters
    model = LassoNetRegressor(
        hidden_dims=(6,3),  # Smaller hidden layer
        verbose=False,  # Disable verbose output
        patience=100,  # Reduced patience
        n_iters=500,  # Fewer iterations
        lambda_start=0.1,  # Start with stronger regularization
        M=10  # Smaller M parameter
    )
    print(f'fitting the model for the variable {target_var} from the variables {parent_vars}')
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = dict(zip(parent_vars, model.feature_importances_))
    
    # Print results
    print(f"Feature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    print(f"Test R2 score: {model.score(X_test, y_test):.4f}")
    print(f"Functional form prediction completed in {time.time() - start_time:.2f} seconds")
    
    return model, feature_importance

def causal_discovery(df, method='SCORE', **kwargs):
    """
    Perform causal discovery on the input data using specified method.
    
    Parameters:
        df (pd.DataFrame): Input data
        method (str): Causal discovery method to use
        **kwargs: Additional arguments to pass to the causal discovery method
        
    Returns:
        tuple: (topological_order, causal_graph, model, feature_importance)
            topological_order: List of variables in topological order
            causal_graph: Learned causal graph
            model: Fitted LassoNet model for the last variable
            feature_importance: Dictionary of feature importance scores
    """
    print("Starting causal discovery pipeline...")
    start_time = time.time()
    
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(X, columns=df.columns)
    
    # Get topological order and causal graph
    topological_order, causal_graph = get_topological_order(df_scaled, method=method, **kwargs)
    
    print(topological_order)
    # Get the last variable and its potential parents
    target_var = topological_order[-1]
    parent_vars = topological_order[:-1]
    print(f'trying to predict functional form of the variable {target_var} from the variables {parent_vars}')
    # Predict functional form of the last variable
    model, feature_importance = predict_functional_form(df_scaled, target_var, parent_vars)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return topological_order, causal_graph, model, feature_importance

# Example usage
if __name__ == "__main__":
    
    relations = [
    lambda X, N: X[:, 0],  # X1 = exogenous (will be overwritten by noise)
    lambda X, N: np.tanh(X[:, 2]) + N,  # X2 = tanh(X3) + N2
    lambda X, N: np.sinc(X[:, 0]) + N,  # X3 = sinc(X1) + N3
    lambda X, N: X[:, 0]**3 - X[:, 0] + N,  # X4 = X1^3 - X1 + N4
    lambda X, N: X[:, 3] * np.sin(X[:, 2]) + N,  # X5 = X4 * sin(X3) + N5
]

    n_samples = 1000
    # For X1, just use noise (exogenous)
    relations[0] = lambda X, N: N  # X1 = N1

    data = generate_scm(n_samples, relations, noise_std=1.0, seed=42)
        # Generate data
        # Perform causal discovery using SCORE method
    topological_order, causal_graph, model, feature_importance = causal_discovery(data, method='SCORE')

    print("\nTopological order:", topological_order)
    print("Causal graph edges:", causal_graph.edges())

    # Example of using different methods
    # DAS method
    # topological_order_das, causal_graph_das, model_das, feature_importance_das = causal_discovery(df, method='DAS')
    
    # NoGAM method
    # topological_order_nogam, causal_graph_nogam, model_nogam, feature_importance_nogam = causal_discovery(df, method='NoGAM')
    
    # CAM method
    # topological_order_cam, causal_graph_cam, model_cam, feature_importance_cam = causal_discovery(df, method='CAM') 