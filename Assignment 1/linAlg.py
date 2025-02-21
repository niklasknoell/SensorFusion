import numpy as np

'''Linear Algebra functions'''
def diag(x):
    # Create diagonal matrix
    return np.diag(x[:, 0])

def sinv(M, M2=0):
    # Stable inverse function
    if np.isscalar(M2) == True:
        M2 = np.eye(len(M))
    return np.linalg.solve(M, M2)

def jitter(M):
    # Add jitter for numerical stabilisation
    return np.eye(len(M)) * 10**-9

def chol(M):
    # Cholesky decomposition + jitter for numerical stabilisation
    return np.linalg.cholesky(M + jitter(M))



'''Linear Algebra matrix reshaping functions'''
def diagdf(df):
    N = int(df.shape[0] / 3)
    blockdf = np.zeros((df.shape[0], df.shape[0]))
    for i in range(N):
        ind1 = i * 3
        ind2 = ind1 + 3
        blockdf[ind1:ind2, ind1:ind2] = df[ind1:ind2, :]
    return blockdf

def matrix3DTo2DHorizontal(M3D):
    Nx, Ny, Nz = M3D.shape
    M2D = np.zeros((Nx, Ny * Nz))
    for kndx in range(0, M3D.shape[2]):
        M2D[:, kndx*Ny:(kndx+1)*Ny] = M3D[:, :, kndx]
    return M2D

def matrix3DTo2DVertical(M3D):
    Nx, Ny, Nz = M3D.shape
    M2D = np.zeros((Nx * Nz, Ny))
    for kndx in range(0, M3D.shape[2]):
        M2D[kndx*Nx:(kndx+1)*Nx, :] = M3D[:, :, kndx]
    return M2D

def matrix3DTo2DDiagonal(M3D):
    Nx, Ny, Nz = M3D.shape
    M2D = np.zeros((Nx * Nz, Ny * Nz))
    for kndx in range(0, M3D.shape[2]):
        M2D[kndx*Nx:(kndx+1)*Nx,  kndx*Ny:(kndx+1)*Ny] = M3D[:, :, kndx]
    return M2D

def blockRotation(Rdata, modelParameters):
    Rho = modelParameters["Rho"]
    k = 0
    A = Rdata.shape[2]
    B = Rho.shape[1]
    dRs = np.zeros((3, 3, A * B))
    for i in range(0, B):
        for j in range(0, A):
            dRs[:, :, k] = crossVector(Rdata[:, :, j] @ Rho[:, i]).T
            k += 1
    return matrix3DTo2DDiagonal(dRs)

def blockDiag(*arrs):
    """Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


'''Data generation helper functions'''
def gridpointsHyperCube(N, D, Din, domain):
    # Create input points on a grid
    X1 = np.linspace(domain[0, 0], domain[0, 1], N)

    if Din == 1:
        X = X1.reshape(D, -1)
    else:
        dict = {}
        inputs = []

        for i in range(0, D):
            inputs.append("x" + str(i))
            dict[inputs[i]] = X1

        X2 = np.array(np.meshgrid(*(v for _, v in sorted(dict.items()))))
        X = np.zeros((D, N**D))
        for i in range(0, D):
            X[i, :] = X2[i, :].reshape(1, N**D)[0, :]

    if Din > D:
        X3 = np.zeros((Din - D, X.shape[1]))
        X = np.vstack((X, X3))
    return X


'''Rotation matrix functions'''
def crossMatrix(a):
    A = np.zeros((3, 3, a.shape[1]))
    for i in range(0, a.shape[1]):
        A[:, :, i] = crossVector(a[:, i])
    return A

def crossVector(a):
    a = a.reshape(-1, 1)
    A = np.zeros((3, 3))
    A[1, 0] = a[2, 0]
    A[2, 0] = -a[1, 0]
    A[2, 1] = a[0, 0]
    A -= A.T
    return A

def so3Rodrigues(psi):
    if psi[0] == 0 and psi[1] == 0 and psi[2] == 0:
        n = psi
        psi_abs = 0
    else:
        psi_abs = np.linalg.norm(psi)
        n = psi / psi_abs
    R = np.zeros((3, 3))
    sin = np.sin(psi_abs)
    cos = np.cos(psi_abs)
    for i in range(0, 3):
        R[i, i] = cos + n[i] ** 2 * (1 - cos)

    R[0, 1] = n[0] * n[1] * (1 - cos) - n[2] * sin
    R[1, 0] = n[0] * n[1] * (1 - cos) + n[2] * sin

    R[0, 2] = n[0] * n[2] * (1 - cos) + n[1] * sin
    R[2, 0] = n[0] * n[2] * (1 - cos) - n[1] * sin

    R[1, 2] = n[1] * n[2] * (1 - cos) - n[0] * sin
    R[2, 1] = n[1] * n[2] * (1 - cos) + n[0] * sin
    return R

def Rx(alpha):
    R = np.eye(3)
    R[1, 1] = np.cos(alpha)
    R[2, 2] = np.cos(alpha)
    R[1, 2] = np.sin(alpha)
    R[2, 1] = -np.sin(alpha)
    return R

def Ry(beta):
    R = np.eye(3)
    R[0, 0] = np.cos(beta)
    R[2, 2] = np.cos(beta)
    R[0, 2] = np.sin(beta)
    R[2, 0] = -np.sin(beta)
    return R

def Rz(gamma):
    R = np.eye(3)
    R[0, 0] = np.cos(gamma)
    R[1, 1] = np.cos(gamma)
    R[0, 1] = -np.sin(gamma)
    R[1, 0] = np.sin(gamma)
    return R

def R2eta(R):
    eta = np.zeros((3, 1))
    # eta[0, 0] = R[2, 1]
    # eta[1, 0] = R[0, 2]
    # eta[2, 0] = R[1, 0]
    if np.array_equal(R, np.eye(3)):
        return eta
    else:
        theta = np.arccos((np.trace(R)-1)/2)
        R2 = theta/(2*np.sin(theta)) * (R - R.T)
        eta[0, 0] = R2[2, 1]
        eta[1, 0] = R2[0, 2]
        eta[2, 0] = R2[1, 0]
        return eta

def expR(eta):
    R = np.eye(3)
    if eta[0] == 0 and eta[1] == 0 and eta[2] == 0:
        return R
    else:
        etaNorm = np.sqrt(eta[0]**2 + eta[1]**2 + eta[2]**2)
        etaCross = crossVector(eta/etaNorm)
        R += np.sin(etaNorm) * etaCross
        R += (1-np.cos(etaNorm)) * etaCross @ etaCross 
        return R

def expQuatLeft(eta, qOld):
    eta1, eta2, eta3 = eta

    # Construct the omega matrix
    omega = np.array([
        [0, eta1, eta2, eta3],
        [-eta1, 0, eta3, -eta2],
        [-eta2, -eta3, 0, eta1],
        [-eta3, eta2, -eta1, 0]
    ])

    etaNorm = np.linalg.norm(eta)
    
    if etaNorm != 0:
        # Calculate the new quaternion
        qNew = (np.cos(etaNorm / 2) * np.eye(4) - (1 / etaNorm) * np.sin(etaNorm / 2) * omega) @ qOld
        # Normalize for numerical stability
        qNew /= np.linalg.norm(qNew)
    else:
        qNew = qOld

    return qNew

def bodyToNavigation(ya, Rba, modelParameters):
    Narray = modelParameters['Narray']
    yb = np.zeros(ya.shape)
    for i in range(Rba.shape[2]):
        yb[:, i*Narray:(i+1)*Narray] = Rba[:, :, i] @ ya[:, i*Narray:(i+1)*Narray] 
    return yb

def navigationToBody(ya, Rba, modelParameters):
    Narray = modelParameters['Narray']
    yb = np.zeros(ya.shape)
    for i in range(Rba.shape[2]):
        yb[:, i*Narray:(i+1)*Narray] = Rba[:, :, i].T @ ya[:, i*Narray:(i+1)*Narray] 
    return yb

def quat2rmat(q):
    """
    Converts quaternions into rotation matrices.
    Assumes that q is Nx4. For a single quaternion, both 1x4 and 4x1 are accepted.
    """
    q = np.asarray(q)
    
    if q.ndim == 1:  # Single quaternion case
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        
        R = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                      [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, 2*q2*q3 - 2*q0*q1],
                      [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 - q1**2 - q2**2 + q3**2]])
    else:  # Multiple quaternions case
        q0, q1, q2, q3 = q[0, :], q[1, :], q[2, :], q[3, :]
        
        R = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                      [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, 2*q2*q3 - 2*q0*q1],
                      [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 - q1**2 - q2**2 + q3**2]])
        
   
    return R


def q2R(q):
    """
    Convert a quaternion into a rotation matrix.

    Parameters:
    quaternion : array-like
        A quaternion represented as a 4-element array or list [q1, q2, q3, q4].

    Returns:
    R : ndarray
        A 3x3 rotation matrix.
    """
    # Normalize the quaternion
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise ValueError("Quaternion is too close to zero.")
    q /= norm  # Normalize

    q1, q2, q3, q4 = q

    # Compute the rotation matrix
    R = np.array([
        [1 - 2*(q3**2 + q4**2), 2*(q2*q3 - q4*q1), 2*(q2*q4 + q3*q1)],
        [2*(q2*q3 + q4*q1), 1 - 2*(q2**2 + q4**2), 2*(q3*q4 - q2*q1)],
        [2*(q2*q4 - q3*q1), 2*(q3*q4 + q2*q1), 1 - 2*(q2**2 + q3**2)]
    ])

    return R



'''Metrics'''
def MSLL(f, P, ftrue, modelParameters):
    theta = modelParameters["theta"]
    MSLL = 0
    ftrue = ftrue.T.reshape(-1, 1)
    f = f.T.reshape(-1, 1)
    for i in range(len(f)):
        MSLL += ((ftrue[i] - f[i]) ** 2 / (P[i, i] + theta[2])) + np.log(2 * np.pi * (P[i, i] + theta[2]))
    return MSLL / (2 * len(f))

def NMSE(f, P, ftrue, modelParameters):
    ftrue = ftrue.T.reshape(-1, 1)
    f = f.T.reshape(-1, 1)
    P = np.diag(P)
    N = len(ftrue)
    return np.sum((ftrue - f) ** 2 / P) / N

def RMSE(fhat, P, ftrue, modelParameters):
    ftrue = ftrue.T.reshape(-1, 1)
    fhat = fhat.T.reshape(-1, 1)
    N = len(ftrue)
    return np.sqrt(np.sum((ftrue - fhat) ** 2) / N)

def MAE(fhat, P, ftrue, modelParameters):
    ftrue = ftrue.T.reshape(-1, 1)
    fhat = fhat.T.reshape(-1, 1)
    N = len(ftrue)
    return np.sum(np.abs(ftrue - fhat)) / N


'''Other'''
def normaliseArray(array):
    minVal = np.min(array)
    maxVal = np.max(array)
    normalisedArray = (array - minVal) / (maxVal - minVal)
    return normalisedArray

def vectorToScalarNorm(y):
    '''Data in 3xN'''
    y = np.sqrt(y[0, :] ** 2 + y[1, :] ** 2 + y[2, :] ** 2)
    return y.reshape(1, -1)

def cov3DTrace(cov):
    n = int(cov.shape[0] / 3)
    traceCov = np.zeros((1, n))
    for i in range(n):
        for j in range(3):
            traceCov[0, i] += cov[3 * i + j, 3 * i + j]
    return traceCov

def cov3DNorm(cov):
    n = int(cov.shape[0] / 3)
    normCov = np.zeros((1, n))
    a = np.zeros((3, 1))
    for i in range(n):
        # for j in range(3):
        #    a[j, 0] = cov[3 * i + j, 3 * i + j]
        a = cov[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)]
        normCov[0, i] += np.linalg.norm(a)
    return normCov

def cov3Dmax(cov):
    n = int(cov.shape[0] / 3)
    maxCov = np.zeros((1, n))
    a = np.zeros((3, 1))
    for i in range(n):
        for j in range(3):
            a[j, 0] = cov[3 * i + j, 3 * i + j]
        maxCov[0, i] = np.max(a)
    return maxCov
