import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_wine, load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SVMVisualizer:
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def load_data(self):
        if self.problem_type == 'classification':
            data = load_wine()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)
        else:
            data = load_diabetes()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train, kernel='rbf'):
        """Train SVM model with specified kernel"""
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.problem_type == 'classification':
            model = SVC(kernel=kernel, random_state=42)
        else:
            model = SVR(kernel=kernel)
            
        model.fit(X_scaled, y_train)
        return model
    
    def plot_feature_importance(self, X_train, y_train, kernel='rbf'):
        """Plot feature correlations and top feature pairs"""
        fig, axes = plt.subplots(1, 2, figsize=(21, 9))
        
        # Feature correlations
        sns.heatmap(X_train.corr(), cmap='RdBu', center=0, fmt='.2f', annot=True, ax=axes[0])
        axes[0].set_title(f'Feature Correlations - {kernel} kernel')

        # Find the top 2 features based on correlation with target
        if self.problem_type == 'classification':
            X_train_pca = self.pca.fit_transform(self.scaler.fit_transform(X_train))
            X_train_plot = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
        else: 
            correlations = [np.corrcoef(X_train[col], y_train)[0,1] for col in X_train.columns]
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Correlation with Target': correlations
            }).sort_values('Correlation with Target', ascending=False)
        
            top_features = feature_importance['Feature'].values[:2]
            X_train_plot = X_train[top_features]

        # Top 2 features relationship
        axes[1].set_title(f'Top Features Relationship - {kernel} kernel')
        if self.problem_type == 'classification':
            sns.scatterplot(data=X_train_plot, 
                          x=X_train_plot.columns[0], 
                          y=X_train_plot.columns[1],
                          sizes = 500, 
                          hue=y_train, ax=axes[1], palette='tab10')
        else:
            sns.scatterplot(data=X_train_plot, 
                          x=X_train_plot.columns[0], 
                          y=X_train_plot.columns[1], 
                          hue=y_train,
                          palette='viridis',
                           ax=axes[1])
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X_train, y_train, kernel='rbf'):
        """Plot decision boundary using PCA for dimensionality reduction"""
        X_scaled = self.scaler.fit_transform(X_train)
        X_pca = self.pca.fit_transform(X_scaled)
        
        model = self.train_model(pd.DataFrame(X_pca), y_train, kernel)
        
        # Create mesh grid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.8)
        plt.colorbar(scatter)
        plt.title(f'Decision Boundary with {kernel} kernel (PCA components)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()
        
    def plot_regression_surface(self, X_train, y_train, kernel='rbf'):
        """Plot regression surface using top 2 PCA components"""
        X_scaled = self.scaler.fit_transform(X_train)
        X_pca = self.pca.fit_transform(X_scaled)
        
        model = self.train_model(pd.DataFrame(X_pca), y_train, kernel)
        
        # Create mesh grid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.6)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y_train, 
                           c=y_train, cmap='viridis')
        plt.colorbar(surf)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Target')
        plt.title(f'Regression Surface with {kernel} kernel (PCA components)')
        plt.show()

    def plot_interactive_regression_surface(self, X_train, y_train, kernel='rbf'):
        """Plot interactive 3D regression surface using top 2 PCA components"""
        x_scaled = self.scaler.fit_transform(X_train)
        X_pca = self.pca.fit_transform(x_scaled)

        model = self.train_model(pd.DataFrame(X_pca), y_train, kernel)

        # Create mesh grid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure(data=[
            go.Surface(z=Z, x=xx, y=yy, colorscale='Viridis', opacity=0.6),
            # Scatter plot of actual data points in 2D
            go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=y_train,
                         mode='markers',
                         marker=dict(size=5, color=y_train, colorscale='Viridis', opacity=0.8)) 
        ])
        fig.update_layout(title=f'Interactive 3D Regression Surface with {kernel} kernel (PCA components)',
                            # make the plot bigger or smaller
                            width=800, height=600,
                          scene=dict(xaxis_title='First Principal Component',
                                     yaxis_title='Second Principal Component',
                                     zaxis_title='Target'))
        fig.show()

    def evaluate_model(self, X_train, X_test, y_train, y_test, kernel='rbf'):
        """Evaluate model performance and print metrics"""
        model = self.train_model(X_train, y_train, kernel)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        if self.problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f'\nResults for {kernel} kernel:')
            print(f'Accuracy: {accuracy:.3f}')
            print(f'Number of support vectors: {model.n_support_.sum()}')
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f'\nResults for {kernel} kernel:')
            print(f'MSE: {mse:.3f}')
            print(f'R2 Score: {r2:.3f}')
            print(f'Number of support vectors: {model.n_support_}')


if __name__ == "__main__":
    # Example usage for classification
    clf_viz = SVMVisualizer(problem_type='classification')
    X_train, X_test, y_train, y_test = clf_viz.load_data()

    # Generate visualizations for classification
    kernels = ['linear', 'rbf']  # modify based on group assignment
    for kernel in kernels:
        print(f"\nAnalyzing {kernel} kernel for classification:")
        clf_viz.plot_feature_importance(X_train, y_train, kernel)
        clf_viz.plot_decision_boundary(X_train, y_train, kernel)
        clf_viz.evaluate_model(X_train, X_test, y_train, y_test, kernel)

    # Example usage for regression
    reg_viz = SVMVisualizer(problem_type='regression')
    X_train, X_test, y_train, y_test = reg_viz.load_data()

    # Generate visualizations for regression
    for kernel in kernels:
        print(f"\nAnalyzing {kernel} kernel for regression:")
        reg_viz.plot_feature_importance(X_train, y_train, kernel)
        reg_viz.plot_regression_surface(X_train, y_train, kernel)
        reg_viz.evaluate_model(X_train, X_test, y_train, y_test, kernel)