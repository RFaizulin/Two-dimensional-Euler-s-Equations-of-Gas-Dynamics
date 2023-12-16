import numpy as np


class RK4Solver:
    def __init__(self, grid_sizes, initial_data, boundary_data, cfl, xa=0, xb=1, ya=0, yb=1):
        self.x_grid_size, self.y_grid_size = grid_sizes
        self.nx, self.ny = self.x_grid_size + 1, self.y_grid_size + 1
        self.x, self.y = np.linspace(xa, xb, self.nx), np.linspace(ya, yb, self.ny)
        self.dx, self.dy = (xb - xa) / self.x_grid_size, (yb - ya) / self.y_grid_size
        self.nx0, self.ny0 = self.nx //2, self.ny // 2
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')

        self.initial_data = initial_data
        self.boundary_data = boundary_data
        self.cfl = cfl
        self.gamma = 1.4
        self.n_eq = 4
        self.n_vars = 4

    def get_xx(self):
        return self.xx

    def get_yy(self):
        return self.yy

    def initial_condition(self):
        V = np.zeros((self.n_vars, self.nx, self.ny), dtype=np.float32)
        for i in range(self.n_vars):
            # Region(1): xa <= x <= xb/2, ya <= y <= yb/2
            V[i][:self.nx0, :self.ny0] = self.initial_data[i][0]
            # Region(2): xb/2 <= x <= xb, ya <= y <= yb/2
            V[i][self.nx0:, :self.ny0] = self.initial_data[i][1]
            # Region(3): xa <= x <= xb/2, yb/2 <= y <= yb
            V[i][:self.nx0, self.ny0:] = self.initial_data[i][2]
            # Region(4): xb/2 <= x <= xb, yb/2 <= y <= yb
            V[i][self.nx0:, self.ny0:] = self.initial_data[i][3]
        return V

    def boundary_condition(self, V):
        for i in range(self.n_vars):
            # Wall(1): y = ya, xa <= x <= xb/2
            V[i][:self.nx0, 0] = self.boundary_data[i][0]
            # Wall(2): y = ya, xb/2 <= x <= xb
            V[i][self.nx0:, 0] = self.boundary_data[i][1]
            # Wall(3): y = yb, xa <= x <= xb/2
            V[i][:self.nx0, -1] = self.boundary_data[i][2]
            # Wall(4): y = yb, xb/2 <= x <= xb
            V[i][self.nx0:, -1] = self.boundary_data[i][3]

            # Wall(5): x = xa, ya <= y <= yb/2
            V[i][0, :self.ny0] = self.boundary_data[i][0]
            # Wall(6): x = xa, yb/2 <= y <= yb
            V[i][0, self.ny0:] = self.boundary_data[i][2]
            # Wall(7): x = xb, ya <= y <= yb/2
            V[i][-1, :self.ny0] = self.boundary_data[i][1]
            # Wall(8): x = xb, yb/2 <= y <= yb
            V[i][-1, self.ny0:] = self.boundary_data[i][3]
        return V

    def transform_V_to_U(self, V):
        U = np.zeros((self.n_eq, self.nx, self.ny), dtype=np.float32)
        U[0] = V[0]
        U[1] = V[0] * V[1]
        U[2] = V[0] * V[2]
        velocity2 = np.square(V[1]) + np.square(V[2])
        U[3] = V[3]/(self.gamma-1) + 0.5*V[0]*velocity2
        return U

    def transform_U_to_V(self, U):
        V = np.zeros((self.n_vars, self.nx, self.ny), dtype=np.float32)
        V[0] = U[0]
        V[1] = U[1] / U[0]
        V[2] = U[2] / U[0]
        velocity2 = np.square(V[1]) + np.square(V[2])
        V[3] = (self.gamma-1) * (U[3] - 0.5*V[0]*velocity2)
        return V

    def timestep(self, V):
        rho, u, v, p = V
        a = np.sqrt(self.gamma * p / rho)
        sn_max = np.max([np.abs(u) + a, np.abs(v) + a])
        dt = self.cfl * self.dx / sn_max
        return dt

    def calculate_D(self, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny), dtype=np.float32)
        if x_derivative:
            D[0, :] = (u[1, :] - u[0, :]) / self.dx
            D[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*self.dx)
            D[-1, :] = (u[-1, :] - u[-2, :]) / self.dx
        else:
            D[:, 0] = (u[:, 1] - u[:, 0]) / self.dy
            D[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*self.dy)
            D[:, -1] = (u[:, -1] - u[:, -2]) / self.dy
        return D

    def calculate_flow(self, xi, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny), dtype=np.float32)
        if x_derivative:
            ur = 0.5*(u[2:, :] + u[1:-1, :])
            ul = 0.5*(u[1:-1, :] + u[:-2, :])
            xi_r_ur = 0.5*xi[1:-1, :]*(ur+np.abs(ur)) + 0.5*xi[2:, :]*(ur-np.abs(ur))
            xi_l_ul = 0.5*xi[:-2, :]*(ul+np.abs(ul)) + 0.5*xi[1:-1, :]*(ul-np.abs(ul))
            D[1:-1, :] = (xi_r_ur - xi_l_ul) / self.dx
        else:
            ur = 0.5*(u[:, 2:] + u[:, 1:-1])
            ul = 0.5*(u[:, 1:-1] + u[:, :-2])
            xi_r_ur = 0.5*xi[:, 1:-1]*(ur+np.abs(ur)) + 0.5*xi[:, 2:]*(ur-np.abs(ur))
            xi_l_ul = 0.5*xi[:, :-2]*(ul+np.abs(ul)) + 0.5*xi[:, 1:-1]*(ul-np.abs(ul))
            D[:, 1:-1] = (xi_r_ur - xi_l_ul) / self.dy
        return D

    def calculate_R(self, U, V):
        U1, U2, U3, U5 = U
        u, v, p = V[1:]

        Dx_p = self.calculate_D(p)
        Dy_p = self.calculate_D(p, x_derivative=False)

        # continuity
        Dx_rho_u = self.calculate_flow(U1, u)

        Dy_rho_v = self.calculate_flow(U1, v, x_derivative=False)

        Dt_rho = -(Dx_rho_u + Dy_rho_v)

        # x momentum
        Dx_rho_uu = self.calculate_flow(U2, u)

        Dy_rho_uv = self.calculate_flow(U2, v, x_derivative=False)

        Dt_rho_u = -(Dx_rho_uu + Dx_p) - Dy_rho_uv

        # y momentum
        Dx_rho_uv = self.calculate_flow(U3, u)

        Dy_rho_vv = self.calculate_flow(U3, v, x_derivative=False)

        Dt_rho_v = -Dx_rho_uv - (Dy_rho_vv + Dy_p)

        # energy
        Dx_Et_u = self.calculate_flow(U5, u)
        Dx_pu = self.calculate_flow(p, u)

        Dy_Et_v = self.calculate_flow(U5, v, x_derivative=False)
        Dy_pv = self.calculate_flow(p, v, x_derivative=False)

        Dt_Et = -(Dx_Et_u + Dx_pu) - (Dy_Et_v + Dy_pv)

        R = np.array([
            Dt_rho,
            Dt_rho_u,
            Dt_rho_v,
            Dt_Et
        ], dtype=np.float32)
        return R

    def rk4_method(self, U, V, dt):
        Rn = self.calculate_R(U, V)
        # step 1
        U1 = U + 0.5*dt*Rn
        R1 = self.calculate_R(U1, V)
        # step 2
        U2 = U + 0.5*dt*R1
        R2 = self.calculate_R(U2, V)
        # step 3
        U3 = U + dt*R2
        R3 = self.calculate_R(U3, V)
        # step 4
        U_new = (U + (Rn + 2*R1 + 2*R2 + R3)*dt/6).astype(dtype=np.float32)
        return U_new

    def solve(self, tf, Vt=dict()):
        n = 0
        if len(Vt) == 0:
            V_old = self.initial_condition()
            t = 0
        else:
            V_old = Vt['V_old']
            t = Vt['t']

        while t <= tf:
            dt = self.timestep(V_old)
            t += dt
            n += 1

            U_old = self.transform_V_to_U(V_old)
            U_new = self.rk4_method(U_old, V_old, dt)
            V_new = self.transform_U_to_V(U_new)
            V_new = self.boundary_condition(V_new)
            V_old = np.copy(V_new)
            if n % 1000 == 0:
                print(t)
                np.savez(f'V_{t}', V_old=V_old, t=t)
            if n % 100 == 0:
                print(n)
        return V_new
    
    
# case 1
grid_sizes = (1500, 1500)
rho_ic = np.array([0.138, 0.5323, 0.5323, 1.5])
u_ic = np.array([1.206, 0, 1.206, 0])
v_ic = np.array([1.206, 1.206, 0, 0])
p_ic = np.array([0.029, 0.3, 0.3, 1.5])
initial_data = np.array([rho_ic, u_ic, v_ic, p_ic])
boundary_data = np.copy(initial_data)
cfl = 0.01
solver = RK4Solver(grid_sizes, initial_data, boundary_data, cfl)
xx, yy = solver.get_xx(), solver.get_yy()
tf = 0.3
V = solver.solve(tf)


# case 2
grid_sizes = (1500, 1500)
rho_ic = np.array([0.8, 1, 1, 0.5313])
u_ic = np.array([0, 0, 0.7276, 0])
v_ic = np.array([0, 0.7276, 0, 0])
p_ic = np.array([1, 1, 1, 0.4])
initial_data = np.array([rho_ic, u_ic, v_ic, p_ic])
boundary_data = np.copy(initial_data)
cfl = 0.01
solver = RK4Solver(grid_sizes, initial_data, boundary_data, cfl, xa=-1, xb=1, ya=-1, yb=1)
xx, yy = solver.get_xx(), solver.get_yy()
tf = 0.52
V = solver.solve(tf)


# case 3
grid_sizes = (1500, 1500)
rho_ic = np.array([0.1379, 0.5323, 0.5323, 1.5])
u_ic = np.array([1.206, 0, 1.206, 0])
v_ic = np.array([1.206, 1.206, 0, 0])
p_ic = np.array([0.029, 0.3, 0.3, 1.5])
initial_data = np.array([rho_ic, u_ic, v_ic, p_ic])
boundary_data = np.copy(initial_data)
cfl = 0.01
solver = RK4Solver(grid_sizes, initial_data, boundary_data, cfl, xa=-1, xb=1, ya=-1, yb=1)
xx, yy = solver.get_xx(), solver.get_yy()
tf = 1.1
V = solver.solve(tf)