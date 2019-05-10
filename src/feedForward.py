'''
def gradient_descent(
        X,
        y,
        theta_0,
        alpha,
        cost_derivate,
        cost,
        treshold = 0.001,
        max_iters=10000):

    theta, last_cost, i = theta_0,sys.maxint, 0
    while i < max_iters and abs(cost(X, y, theta) - last_cost) > treshold:
        last_cost = cost(X, y, theta)
        theta -= alpha * cost_derivate(theta, X,y)
        i +=1

        return theta

def linear_cost(theta, X, y):
        m, n = X.shape
        h = np.matmul(X, theta)
        sq = (y - h) ** 2

        return sq.sum() / (2 * m)

    Examen Corto

def linear_cost_derivado(theta, X, y):
        m, n = X.shape
        h = np.matmul(X, theta)
        sq = (h * y)
        return np.matmul(sq, sum(), X) / m
'''