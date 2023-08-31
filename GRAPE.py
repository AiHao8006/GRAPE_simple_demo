'''
A demo of GRAPE algorithm for quantum gates.
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm


class GRAPE():
    def __init__(self):
        # define the Pauli matrices
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])

        # Hamiltonian: H = w0 * Z + w1 * X
        # H0 = 0, which means resonance in a rotating frame

        # start gate
        self.U0 = self.I
        # traget gate, if global phase is not eliminated, a quantum gate should be in SU group with det(U) = 1
        self.U_F = expm(1j * self.Y)

        # time list
        self.t_list = np.linspace(0, 1, 101)
        self.dt = self.t_list[1] - self.t_list[0]
        # pulses, which are dynamically changing during the algorithm, and the initial guesses are below
        # self.w0 = np.sin(np.pi * self.t_list)
        # self.w1 = np.cos(np.pi * self.t_list)
        self.w0 = np.random.randn(len(self.t_list))
        self.w1 = np.random.randn(len(self.t_list))
        # the limit of pulses are [-2,2]
        self.w0 = np.clip(self.w0, -2, 2)
        self.w1 = np.clip(self.w1, -2, 2)
        # fidelity corresponding to the pulses
        self.fidelity = None
        
    def U_indt(self):
        '''
        get the list of single-step evolution operators,
        i.e. U(t_n, t_n+1) = exp(-i * H * dt)
        '''
        U_indt_list = []
        for i in range(len(self.t_list)):
            U_indt = expm(-1j * (self.w0[i] * self.Z + self.w1[i] * self.X) * self.dt)
            U_indt_list.append(U_indt)
        return U_indt_list

    def bidirect_evolu(self):
        '''
        get the lists of evolution operators from U0 to U_F and from U_F to U0
        '''
        U_indt_list = self.U_indt()

        A_list = []
        A = self.U0
        for i in range(len(self.t_list)):
            A = np.dot(U_indt_list[i], A)
            A_list.append(A)
        
        B_list = []
        B = self.U_F
        for i in range(len(self.t_list)):
            B = np.dot(U_indt_list[-i-1].conj().T, B)
            B_list.append(B)
        B_list.reverse()

        return A_list, B_list
    
    @staticmethod
    def inner_product(A, B):
        '''
        according to Ref. srep36090 (2016),
        the inner product only calculate the trace of A^dagger * B
        '''
        return np.trace(np.dot(A.conj().T, B))
    
    def iteration_onestep(self, lr=0.1):
        '''
        iteration of GRAPE algorithm
        '''
        partial_w0 = np.zeros_like(self.w0)
        partial_w1 = np.zeros_like(self.w1)

        A_list, B_list = self.bidirect_evolu()

        for i in range(len(self.t_list)):
            partial_w0[i] = -2 * np.real(self.inner_product(B_list[i], 1j * self.dt * self.Z @ A_list[i]) * self.inner_product(A_list[i], B_list[i]))
            partial_w1[i] = -2 * np.real(self.inner_product(B_list[i], 1j * self.dt * self.X @ A_list[i]) * self.inner_product(A_list[i], B_list[i]))
        
        self.w0 = self.w0 + lr * partial_w0
        self.w1 = self.w1 + lr * partial_w1

        # the limit of pulses are [-2,2]
        self.w0 = np.clip(self.w0, -2, 2)
        self.w1 = np.clip(self.w1, -2, 2)

        self.fidelity = np.abs(self.inner_product(A_list[-1], self.U_F)) ** 2 / 4
        return self.fidelity
    
    def PWC_pulse(self, pwc_pulse):
        '''
        get the piecewise constant pulse, then use plt.plot to plot it
        '''
        pwc_pulse = np.insert(pwc_pulse, 0, 0)
        pwc_pulse = np.append(pwc_pulse, 0)
        time_steps = np.arange(0, len(pwc_pulse)) * self.dt
        time_steps_stair = np.repeat(time_steps, 2)[1:-3]
        pwc_pulse_stair = np.repeat(pwc_pulse, 2)[2:-2]

        return time_steps_stair, pwc_pulse_stair
    

if __name__ == '__main__':
    G = GRAPE()

    fid = []
    i = 0
    while i < 1501:
        fidelity = G.iteration_onestep(lr=0.5)
        print('{}-th\t fidelity: {:4f}'.format(i, fidelity))

        fid.append(fidelity)

        if i % 10 == 0:
            plt.clf()
            time_steps_stair0, pwc_pulse_stair0 = G.PWC_pulse(G.w0)
            plt.plot(time_steps_stair0, pwc_pulse_stair0, 'b-')
            time_steps_stair1, pwc_pulse_stair1 = G.PWC_pulse(G.w1)
            plt.plot(time_steps_stair1, pwc_pulse_stair1, 'r-')
            plt.xlabel('time')
            plt.ylabel('pulse strength')
            plt.title('{}-th fidelity: {:4f}'.format(i, fidelity))
            plt.pause(0.01)

        i += 1

    plt.show()

    plt.plot(fid)
    plt.xlabel('iteration')
    plt.ylabel('fidelity')
    plt.show()
