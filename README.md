# Principal Component Analysis via Singular Value Decomposition

## 📊 Project Overview

The goal of this project is to implement **Principal Component Analysis (PCA)** by explicitly utilizing **Singular Value Decomposition (SVD)**. The project demonstrates the complete process: constructing the data manifold, decomposing the matrix into singular vectors, and projecting high-dimensional data onto a lower-dimensional subspace defined by orthogonal axes of maximum variance.

---

## 🧠 Theoretical Background

### The Linear Algebra Formulation

PCA is strictly a basis transformation. The objective is to rotate the coordinate system of the data such that the new axes, which correspond to the principal components, align with the directions of greatest variance.

We define our dataset as a matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$:
* $N$: Number of samples.
* $D$: Number of features.

We treat the data as a collection of $N$ points in a $D$-dimensional vector space.

### 1. The Centered Data Matrix

The SVD must be performed on the **centered** data matrix. We translate the data cloud so its centroid lies at the origin.

Let $\mathbf{x}_j$ be the column vector representing the $j$-th feature. We compute the mean $\mu_j$ for each feature:

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} X_{ij}
$$

The centered matrix $\mathbf{X}_c$ is constructed by subtracting these means from every entry:

$$
\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mathbf{\mu}^T = 
\begin{bmatrix}
x_{11} - \mu_1 & x_{12} - \mu_2 & \dots & x_{1D} - \mu_D \\
x_{21} - \mu_1 & x_{22} - \mu_2 & \dots & x_{2D} - \mu_D \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} - \mu_1 & x_{N2} - \mu_2 & \dots & x_{ND} - \mu_D
\end{bmatrix}
$$

### 2. Singular Value Decomposition

The SVD decomposes the centered data matrix into three constituent matrices that isolate the geometric structure of the data:

$$
\mathbf{X}_c = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

#### The Left Singular Vectors ($\mathbf{U}$)
$\mathbf{U}$ is an $N \times N$ orthogonal matrix satisfying $\mathbf{U}^T \mathbf{U} = \mathbf{I}_N$. The columns of $\mathbf{U}$, denoted as $\mathbf{u}_i$, form an orthonormal basis for the sample space.

We can visualize $\mathbf{U}$ as a collection of column vectors:

$$
\mathbf{U} = 
\begin{bmatrix}
| & | & & | \\
\mathbf{u}_1 & \mathbf{u}_2 & \dots & \mathbf{u}_N \\
| & | & & |
\end{bmatrix}
$$

The orthogonality property implies that the dot product of any two distinct columns is zero, and the dot product of a column with itself is one:

$$
\mathbf{u}_i^T \mathbf{u}_j = 
\begin{cases} 
1 & \text{if } i=j \\
0 & \text{if } i \neq j 
\end{cases}
$$

#### The Singular Values ($\mathbf{\Sigma}$)
$\mathbf{\Sigma}$ is an $N \times D$ rectangular diagonal matrix containing the **Singular Values** ($\sigma_i$).

$$
\mathbf{\Sigma} = 
\begin{bmatrix}
\sigma_1 & 0 & \dots & 0 \\
0 & \sigma_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \sigma_D \\
0 & 0 & \dots & 0
\end{bmatrix} \quad \text{(Assuming } N > D \text{)}
$$

The values are non-negative real numbers sorted in descending order: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_D \ge 0$. These values represent the magnitude of the variance along each principal component axis.

#### The Right Singular Vectors ($\mathbf{V}^T$)
$\mathbf{V}^T$ is a $D \times D$ orthogonal matrix satisfying $\mathbf{V}^T \mathbf{V} = \mathbf{I}_D$. The rows of $\mathbf{V}^T$, denoted as $\mathbf{v}_i$, are the **Principal Components**. These vectors define the new orthonormal basis for the feature space.

We can visualize $\mathbf{V}^T$ as a stack of row vectors, where each row represents a principal axis:

$$
\mathbf{V}^T = 
\begin{bmatrix}
\text{---} & \mathbf{v}_1^T & \text{---} \\
\text{---} & \mathbf{v}_2^T & \text{---} \\
 & \vdots & \\
\text{---} & \mathbf{v}_D^T & \text{---}
\end{bmatrix}
$$

Similar to $\mathbf{U}$, the orthogonality property ensures that these principal component vectors are uncorrelated and normalized:

$$
\mathbf{v}_i^T \mathbf{v}_j = 
\begin{cases} 
1 & \text{if } i=j \\
0 & \text{if } i \neq j 
\end{cases}
$$

### 3. Connection to Covariance and Eigenvalues

Classical PCA often solves for the eigenstructure of the Covariance Matrix $\mathbf{C}$. When deriving it, we utilize the normalizing factor $\frac{1}{N-1}$ rather than $\frac{1}{N}$. This technique, known as **Bessel's Correction**, ensures an unbiased estimator of the population variance when working with a finite sample.

$$
\mathbf{C} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c
$$

Substituting the SVD of $\mathbf{X}_c$ into this equation reveals the direct mathematical link:

$$
\mathbf{C} = \frac{1}{N-1} (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)^T (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)
$$

Using the property $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$:

$$
\mathbf{C} = \frac{1}{N-1} \mathbf{V} \mathbf{\Sigma}^T \underbrace{\mathbf{U}^T \mathbf{U}}_{\mathbf{I}} \mathbf{\Sigma} \mathbf{V}^T
$$

$$
\mathbf{C} = \mathbf{V} \left( \frac{\mathbf{\Sigma}^2}{N-1} \right) \mathbf{V}^T
$$

This derivation proves that:
1.  The Principal Components are exactly the columns of $\mathbf{V}$.
2.  The Eigenvalues $\lambda_i$ of $\mathbf{C}$ are derived from the singular values by:

    $$
    \lambda_i = \frac{\sigma_i^2}{N-1}
    $$

---

## ⚙️ Application

### 1. Dimension Selection

**Mathematical Formulation**<br>
Once $\mathbf{\Sigma}$ is computed via SVD, we quantify the information content to determine how many dimensions are necessary to represent the data.

The proportion of total variance captured by the $k$-th component is:

$$
\eta_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j} = \frac{\sigma_k^2}{\sum_{j=1}^D \sigma_j^2}
$$

The **Cumulative Explained Variance** for the top $K$ components is:

$$
H_K = \sum_{k=1}^K \eta_k
$$

### 2. Subspace Projection

**Mathematical Formulation**<br>
To perform dimensionality reduction, we retain only the basis vectors associated with significant variance. We form a projection matrix $\mathbf{W}_K$ using the first $K$ columns of $\mathbf{V}$.

$$
\mathbf{W}_K = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_K] \in \mathbb{R}^{D \times K}
$$

The original high-dimensional centered data $\mathbf{X}_c$ is projected onto this subspace via matrix multiplication:

$$
\mathbf{T}_K = \mathbf{X}_c \mathbf{W}_K
$$

The resulting matrix $\mathbf{T}_K$ is $N \times K$. It contains the the coordinates of the original data points in the new reduced principal component space.

---

## ⚠️ Limitations

| Limitation | Description |
| :--- |:---|
| **Linearity Assumption** | PCA relies on matrix operations that define linear hyperplanes. It cannot effectively compress data lying on non-linear manifolds without kernel methods. |
| **Outlier Sensitivity** | The use of mean centering and variance makes PCA sensitive to outliers. Extreme values can pull the principal components away from the true direction of maximum variance. |
| **Orthogonality Constraint** | PCA enforces that principal components must be orthogonal to each other. In some physical or biological systems, the true underlying factors generating the data might be correlated and thus non-orthogonal. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/principal-component-analysis-via-singular-value-decomposition.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis**:
    ```bash
    jupyter notebook
    ```

---

## 📂 Project Structure

* `data_generation.py`: A script to generate synthetic multivariate data.
* `main.ipynb`: The primary notebook for analysis, and plotting.
* `requirements.txt`: Python dependencies.
* `README.md`: Project documentation.