def get_logistic_regression_latex():
    return {
        "linear_model": r"z = w^\top x + b",
        "sigmoid": r"\sigma(z) = \frac{1}{1 + e^{-z}}",
        "probability": r"p(y=1|x) = \sigma(w^\top x + b)",
        "log_likelihood": r"\mathcal{L}(w,b) = \sum_{i=1}^N \left[ y_i \log p_i + (1-y_i) \log (1-p_i) \right]",
        "loss": r"J(w,b) = -\frac{1}{N} \mathcal{L}(w,b)",
        "gradient": r"\frac{\partial J}{\partial w} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)x_i",
        "update_rule": r"w \leftarrow w - \eta \nabla_w J"
    }

def get_decision_tree_latex():
    return {
        "entropy": r"H(S) = -\sum_{k} p_k \log_2(p_k)",
        "gini": r"G(S) = 1 - \sum_{k} p_k^2",
        "information_gain": r"IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)"
    }

def get_metrics_latex():
    return {
        "accuracy": r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
        "precision": r"\text{Precision} = \frac{TP}{TP + FP}",
        "recall": r"\text{Recall} = \frac{TP}{TP + FN}",
        "f1": r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}",
        "roc_auc": r"AUC = \int_0^1 TPR(FPR^{-1}(u)) \, du"
    }
