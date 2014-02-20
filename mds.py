from __future__ import division
import numpy as np

def mds(D, Q=2, Dsup=None, fixsigns=True):
    """
    Given a distance matrix computes an MDS projection into Q dimensions.

    Parameters
    ----------
    D     The matrix of distances between individuals.
    Q     The size of the low-dimensional space.
    Dsup  A matrix of "out-of-sample" distances.  
          If D is N by N, then Dsup should be N by Nsup, where Nsup
          is the number of out of sample points distances.
    fixsigns  
         The signs of the eigenvectors are arbitrary.  If fixsigns is true
         the projections are multiplied by -1 if necessary to ensure that
         the first element of the projection has a positive component for
         each eigenvector.

    Returns
    -------
    A tuple of the low dimensional embedding, the eigenvalues of
    the decomposition and the coordinates of the out of sample points if
    any are given.

    Notation follows "Metric Multidimensional Scaling (MDS): Distance
    Matrices" by Herve Abdi in Neil Salkind (Ed.) (2007). Encyclopedia of
    Measurement and Statistics. Thousand Oaks (CA): Sage.
    http://www.utd.edu/~herve/Abdi-MDS2007-pretty.pdf
    """
    N = D.shape[0]

    m = np.ones(N)/N
    Xi = np.eye(N) - np.ones((N,N))/N
    S = -0.5 * np.dot(Xi, np.dot(D, Xi.T))
    lam, U = np.linalg.eig(S)

    I, = np.nonzero(lam < 10e-8)
    lam = np.delete(lam, I)
    U = np.delete(U, I, axis=1)
    U = np.real(U)
    I = np.argsort(1/lam)
    lam = lam[I]
    U = U[:,I]
    # Originally: F = dot(inv(sqrt(diag(m))), dot(U, sqrt(diag(lam))))
    F = np.dot(np.diag(1/np.sqrt(m)), np.dot(U, np.diag(np.sqrt(lam))))/np.sqrt(N)
    Q = min(Q, len(I))
    if fixsigns:
        for i in range(Q):
            if F[0,i] < 0:
                F[:,i] *= -1
    F[:,1:] *= -1
    if Dsup is None:
        return F[:,:Q], lam
    
    Nsup = Dsup.shape[1]                 # Number of out of sample vectors
    Ssup = -np.dot(Xi, Dsup - np.outer(np.dot(D, m), np.ones(Nsup)))/2
    Fsup = np.dot(Ssup.T, np.dot(F, np.diag(1/lam)))
    return F[:,:Q], lam, Fsup[:,:Q]

if __name__ == "__main__":
    from matplotlib.pyplot import *
    from matplotlib.pylab import *

    def trymds(N=20, M=8, q=2):
        """
        Try mds with N data points in M dimensions, where the real dimension of
        the data is q.
        """
        # Generate N points on a circle/sphere centred on the origin
        X = randn(N, q)
        for x in X:
            x /= norm(x)

        # Supplementary points along the coordinate axes
        E = eye(q)
        Xsup = E.copy()
        for s in linspace(0.1, 0.9, 5):
            Xsup = vstack((Xsup, E*s))


        # Random projection into M dimensions.
        A = randn(M, q)
        X = dot(X, A.T)
        X += randn(N,M)*1e-2     # Add a bit of noise to prevent exactly zero eigenvalues
        Xsup = dot(Xsup, A.T)
        Nsup = len(Xsup)
        Xsup += randn(Nsup, M)*1e-2


        # Get the distances
        D = zeros((N, N))
        for i in range(N):
            for j in range(i):
                D[i,j] = norm(X[i]-X[j])**2
                D[j,i] = D[i,j]

        Dsup = zeros((N, Nsup))
        for i in range(N):
            for j in range(Nsup):
                Dsup[i,j] = norm(X[i]-Xsup[j])**2

        Z, evals, Zsup = mds(D, q, Dsup)


        dfig = figure()
        subplot(2,3,1)
        imshow(D)
        title("D")
        colorbar()

        Dapprox = zeros((N, N))
        for i in range(N):
            for j in range(N):
                Dapprox[i,j] = norm(Z[i]-Z[j])**2
                Dapprox[j,i] = Dapprox[i,j]

        Dsupa = zeros((N, Nsup))
        for i in range(N):
            for j in range(Nsup):
                Dsupa[i,j] = norm(Z[i]-Zsup[j])**2

        subplot(2,3,2)
        imshow(Dapprox)
        colorbar()
        title("Approximate D")

        subplot(2,3,4)
        semilogy(arange(len(evals))+1, evals, 'b-+')

        subplot(2,3,5)
        plot(Z[:,0], Z[:,1], 'r+')
        axis("scaled")
        hold(True)
        plot(Zsup[:,0], Zsup[:,1], 'b+')


        subplot(2,3,3)
        imshow(Dsup)
        colorbar()
        title("Dsup")

        subplot(2,3,6)
        imshow(Dsupa)
        colorbar()
        title("Approximate Dsup")

        show(block=True)
        
    trymds(q=3)
