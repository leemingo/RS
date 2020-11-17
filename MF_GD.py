import numpy as np


class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
                :param R: rating matrix
                :param k: latent parameter
                :param learning_rate: alpha on weight update
                :param reg_param: beta on weight update
                :param epochs: training epochs
                :param verbose: print status
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        self._P = np.random.normal(size = (self._num_users, self._k))
        self._Q = np.random.normal(size = (self._num_items, self._k))

        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        self._training_process = []
        for epoch in range(self._epochs):
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print('Iteration: %d ; cost = %4.f' % (epoch+1, cost))

    def cost(self):
        '''
        computer root mean square error
        :return: rmse cost
        '''


        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)

    def gradient(self, error, i, j):
        '''
        param error: rating - prediction error
        param i: user index
        param j: item index
        :return: gradient of latent feature tuple
        '''

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])

        return dp, dq

    def gradient_descent(self, i, j, rating):
        '''
        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i, j)
        '''

        prediction = self.get_prediction(i, j)
        error = rating - prediction

        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        '''

        :return: prediction of r_ij
        '''

        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

    def print_results(self):
        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs - 1][1])



if __name__ == '__main__':
    R = np.array(
        [
            [1, 0, 0, 1, 3],
            [2, 0, 3, 1, 1],
            [1, 2, 0, 5, 0],
            [1, 0, 0, 4, 4],
            [2, 1, 5, 4, 0],
            [5, 1, 5, 4, 0],
            [0, 0, 0, 1, 0],
        ]
    )

    factorizer = MatrixFactorization(R, k = 3, learning_rate= 0.01, reg_param= 0.01, epochs = 300, verbose = True)
    factorizer.fit()
    factorizer.print_results()