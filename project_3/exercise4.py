import numpy
import matplotlib.pyplot as plt
import dataclasses


@dataclasses.dataclass
class Flags:
    problem_a_b = False
    problem_c = True



def matrix_factorization(R,P,Q,K,steps = 5000,alpha = 0.002, beta = 0.02):
    Q = Q.T
    e = []
    for a in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0 :
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        error = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    error = error + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]),2)
                    for k in range(K):
                        error = error + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        e.append(error)
        if error < 0.001:
            break
    return P,Q.T,e


R = [[1,0,2,0,0,1],
     [0,0,4,2,0,0],
     [3,5,0,4,4,3],
     [0,4,1,0,3,0],
     [0,0,2,5,4,3],
     [5,0,0,0,2,0],
     [0,4,3,0,0,0],
     [0,0,0,4,0,2],
     [5,0,4,0,0,0],
     [0,2,3,0,0,0],
     [4,1,5,2,2,4],
     [0,3,0,0,5,0],
     ]

R = numpy.array(R)
N = len(R)
M = len(R[0])
K = 3

# (a)
if Flags.problem_a_b:
    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ , e = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)



# (b)
    plt.scatter(range(len(e)), e, s = 10, color = 'green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Interation')
    plt.ylabel('Squared Error')
    plt.title('Error Plot')
    plt.show()
    print(nR)
    print(nR[4,0])


# (c)
# Select n learning rates using the grid search method, plot the squared errors for each learning rate, and observe which curve converges the fastest. Finally, identify the learning rate (alpha) that leads to the fastest convergence
if Flags.problem_c:
    error_lists = []

    alpha = numpy.linspace(0.001, 0.1, 5)
    for a in alpha:
        P = numpy.random.rand(N, K)
        Q = numpy.random.rand(M, K)
        nP, nQ, error = matrix_factorization(R, P, Q, K)
        error_lists.append(error)

    for a, error in zip(alpha, error_lists):
        plt.plot(error, label=f'lr={a}')

    plt.xlabel('Iterations')
    plt.ylabel('Squared Error')
    plt.title('Error versus Iterations for different learning rates')
    plt.legend()
    plt.show()