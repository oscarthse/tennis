import numpy as np
import matplotlib.pyplot as plt

def plot_l2_circle():
    w1 = np.linspace(-2, 2, 400)
    w2 = np.linspace(-2, 2, 400)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss Function (Elliptical) centered at (1.5, 1.5)
    J = (W1 - 1.5)**2 + 2 * (W2 - 1.5)**2

    # Constraint (Circle)
    Constraint = W1**2 + W2**2

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Loss Contours
    ax.contour(W1, W2, J, levels=[0.5, 1.5, 3, 5, 8], colors='blue', alpha=0.6)

    # Plot Constraint Region
    circle = plt.Circle((0, 0), 1, color='red', alpha=0.3, label=r'Constraint $\sum w^2 \leq C$')
    ax.add_patch(circle)

    # Plot Tangency Point (Approximate)
    # For circle radius 1, tangency is roughly at intersection
    ax.scatter([0.6], [0.8], color='black', zorder=10, label='Optimal Solution')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('L2 Regularization (Ridge)')
    ax.legend()
    plt.savefig('src/dashboard/pages/l2_circle.png', dpi=100)
    plt.close()

def plot_l1_diamond():
    w1 = np.linspace(-2, 2, 400)
    w2 = np.linspace(-2, 2, 400)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss Function (Elliptical) centered at (1.5, 1.5)
    J = (W1 - 1.5)**2 + 2 * (W2 - 1.5)**2

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Loss Contours
    ax.contour(W1, W2, J, levels=[0.5, 1.5, 3, 5, 8], colors='blue', alpha=0.6)

    # Plot Constraint Region (Diamond)
    # Diamond is defined by |x| + |y| <= 1
    diamond_x = [1, 0, -1, 0, 1]
    diamond_y = [0, 1, 0, -1, 0]
    ax.fill(diamond_x, diamond_y, color='green', alpha=0.3, label=r'Constraint $\sum |w| \leq C$')

    # Plot Tangency Point (Corner)
    ax.scatter([0], [1], color='black', zorder=10, label='Optimal Solution (Sparse)')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('L1 Regularization (Lasso)')
    ax.legend()
    plt.savefig('src/dashboard/pages/l1_diamond.png', dpi=100)
    plt.close()

if __name__ == "__main__":
    plot_l2_circle()
    plot_l1_diamond()
    print("Images generated successfully.")
