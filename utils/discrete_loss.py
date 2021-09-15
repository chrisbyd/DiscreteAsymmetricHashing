import torch


def solve_dcc(W, Y, U, B, eta =1e-2, mu = 1e-2):
    """
    DCC.
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * U

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B
