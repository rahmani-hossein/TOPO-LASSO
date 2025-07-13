import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lassonet import LassoNetRegressor, plot_path, LassoNetRegressorCV

# --- 1. Data Generation ---
np.random.seed(42)
n_samples = 500
n_features = 10
n_informative = 2

X = np.random.rand(n_samples, n_features) * 10 - 5 # Features in [-5, 5]
y_true = np.sin(np.pi * X[:, 0] / 5) + (X[:, 1] / 3)**2
noise_level = 0.5
y = y_true + noise_level * np.random.randn(n_samples)
y = y.reshape(-1, 1) # Reshape y to be a column vector for LassoNet

# --- 2. Data Preprocessing ---
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y) # Also scale y for better NN performance

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
y_train = y_train.ravel() # LassoNetRegressor expects 1D y for path/fit
y_test = y_test.ravel()

# --- 3. Model Definition and Path Computation ---
print("Training LassoNetRegressor and computing path...")
model = LassoNetRegressor(
    hidden_dims=(64,),    # Single hidden layer with 64 neurons
    lambda_start='auto',  # Automatically determine starting lambda
    path_multiplier=1.02, # Controls how fast lambda decreases
    patience=10,          # For early stopping of internal epochs
    random_state=42,
    verbose=0             # Set to 1 for more output
)

# The.path() method computes the sequence of models
# return_state_dicts=True allows loading specific models from the path
path_info = model.path(X_train, y_train, X_val=X_test, y_val=y_test, return_state_dicts=True)

print(f"Path computed. Number of models in path: {len(path_info)}")

# The path_info is a list of objects, each with attributes like:
# item.lambda_val: lambda value for this model
# item.n_selected_features: number of features selected
# item.state_dict: parameters of the model
# item.val_score (if X_val, y_val provided): score on validation set

# Example: Print info for the first few models in the path
for i, item in enumerate(path_info[:5]):
    print(dir(item))
    # or
    print(item.__dict__)
    # Change item.lambda_val to item.lambda_
    num_selected = np.sum(item.selected) if hasattr(item.selected, 'sum') else len(item.selected)
    print(f"Model {i}: Lambda={item.lambda_:.4f}, Features Selected={num_selected}, Val loss={item.val_loss if hasattr(item, 'val_loss') else 'N/A'}")
# --- Visualization (discussed in Section 5.2) ---
# plot_path requires the model to have.n_features_ and.feature_importances_
# these are set during.path().
# The function also needs test data to plot performance.
print("\nPlotting regularization path...")
plot_path(model, X_test, y_test)
plt.title("LassoNet Regularization Path (Test R-squared vs. # Features)")
plt.savefig("lassonet_nonlinear_path.png")
plt.show()

# --- Select a model from the path (e.g., based on plot_path or a specific number of features) ---
# For example, let's find a model with a certain number of features or best validation score
# This is a manual selection process if not using LassoNetRegressorCV
# Here, we'll pick a model from the path, e.g., one with a good balance or a specific number of features
# For demonstration, let's pick a model that selected more than 0 but not all features, if available
# A more systematic way is to use plot_path to guide selection or use LassoNetRegressorCV

# Find best model based on validation score from path_info (if val data was provided to.path)
if hasattr(path_info, 'val_score'):
    best_item_on_path = max(path_info, key=lambda item: item.val_score if item.val_score is not None else -np.inf)
    print(f"\nBest model on path (based on val_score): Lambda={best_item_on_path.lambda_:.4f}, Features={np.sum(best_item_on_path.selected) if hasattr(best_item_on_path.selected, 'sum') else len(best_item_on_path.selected)}, Val Score={best_item_on_path.val_score:.4f}")
    model.load(best_item_on_path.state_dict)
else: # Fallback: pick a model towards the sparser end that still has some features
    selected_item = None
    for item in reversed(path_info): # Iterate from sparse to dense
        if item.n_selected_features > 0 and item.n_selected_features < n_features:
            selected_item = item
            break
    if selected_item:
        print(f"\nSelected model from path: Lambda={selected_item.lambda_val:.4f}, Features={selected_item.n_selected_features}")
        model.load(selected_item.state_dict)
    else: # If no such model, load the densest one (last in path_info before all features are gone)
        model.load(path_info.state_dict) # Load the first model (densest)
        print(f"\nLoaded densest model from path: Lambda={path_info.lambda_val:.4f}, Features={path_info.n_selected_features}")


# Make predictions with the loaded model
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel() # y_test was also scaled

r2 = r2_score(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)

print(f"\nPerformance of selected model from path:")
print(f"  R-squared: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Number of selected features: {model.n_selected_features_}")
print(f"  Selected feature indices: {np.where(model.feature_importances_ > 0)}") # Features with non-zero importance