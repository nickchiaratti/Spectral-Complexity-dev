import numpy as np; np.random.seed(0); N=10000; L=3; Y = np.random.randn(L, N); Y[1] += 0.9*Y[0]; Y[2] += 0.9*Y[1]; RR = np.dot(Y, Y.T); RRi = np.linalg.pinv(RR + 1e-6*np.eye(L)); w = np.zeros((L, N));
for i in range(L):
    col = RRi[:, i:i+1]
    row = RRi[i:i+1, :]
    XX = RRi - np.dot(col, row) / RRi[i, i]
    RRa = RR[:, i].copy()
    RRa[i] = 0.0
    beta = np.dot(XX, RRa)
    beta[i] = 0.0
    w[i, :] = Y[i, :] - np.dot(beta, Y)
cov_w = np.dot(w, w.T)/N; print('Full w cov:\n', cov_w); print('True cov:\n', np.cov(Y))
