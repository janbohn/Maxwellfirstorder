from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
#from fenics import *
#from fenics_rhs import FenicsRHS
#import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import gmres
from bempp.api.fenics_interface import FenicsOperator
from bempp.api import as_matrix
from haha import *
from bempp.api.operators.boundary import maxwell
import operatorn
import timeit
import sys
sys.setrecursionlimit(100000)
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

def Maxwellfuncunsym(T,N,tau,h,m,rho,L,tolgmres,eps,mu,sig,theta,alpha,Ce ,X,Y,trace_space,trace_matrix):#,J,m0,E0,H0):
    print('h is ', h, ' N is', N)
    start = timeit.default_timer()
    #bempp.api.global_parameters.assembly.boundary_operator_assembly_type = "dense"
    # OrderQF = 1
    # tol = 10 ** (-8)
    # bempp.api.global_parameters.hmat.eps = 10 ** -1
    # bempp.api.global_parameters.hmat.max_block_size=1000000
    # bempp.api.global_parameters.hmat.max_rank = 10
    # bempp.api.global_parameters.hmat.min_block_size = 500
    # bempp.api.global_parameters.hmat.cutoff = 1.7*10**100
    # bempp.api.global_parameters.assembly.interpolation_points_per_wavelength=100
    # bempp.api.global_parameters.quadrature.near.max_rel_dist = 1
    # bempp.api.global_parameters.quadrature.near.single_order = OrderQF
    # bempp.api.global_parameters.quadrature.near.double_order = OrderQF
    #
    # bempp.api.global_parameters.quadrature.medium.max_rel_dist = 1
    # bempp.api.global_parameters.quadrature.medium.single_order = OrderQF
    # bempp.api.global_parameters.quadrature.medium.double_order = OrderQF
    #
    # bempp.api.global_parameters.quadrature.far.single_order = OrderQF
    # bempp.api.global_parameters.quadrature.far.double_order = OrderQF
    #
    # bempp.api.global_parameters.quadrature.double_singular = OrderQF
    timeoutput = True  # time measurements
    plotoutput = False
    #meshh = UnitCubeMesh(h, h, h)
    # spaces
    #Xr = FiniteElement("N1curl", meshh.ufl_cell(), 1)
    #X = FunctionSpace(meshh, Xr)
    #fenics_space = X
    #Yr = VectorElement("DG", meshh.ufl_cell(), 0)
    #Y = FunctionSpace(meshh, Yr)

    #trace_space, trace_matrix = n1curl_to_rt0_tangential_trace(fenics_space)  # trace space and restriction matrix

    bc_space = bempp.api.function_space(trace_space.grid, "BC", 0)  # domain spaces
    rwg_space = bempp.api.function_space(trace_space.grid, "B-RWG", 0)
    #snc_space = bempp.api.function_space(trace_space.grid, "B-SNC", 0)  # dual to range spaces
    rbc_space = bempp.api.function_space(trace_space.grid, "RBC", 0)
    brt_space = bempp.api.function_space(trace_space.grid, "B-RT", 0)

    nBEM = trace_space.global_dof_count  # DOFs
    nFEME = X.dim()
    nFEMH = Y.dim()

    class MyExpressionccE(UserExpression):
        def eval(self, value, x):
            s1 = sin(np.pi * x[0])
            s2 = sin(np.pi * x[1])
            s3 = sin(np.pi * x[2])
            s1s = s1 ** 2
            s2s = s2 ** 2
            s3s = s3 ** 2
            c1 = cos(np.pi * x[0])
            c2 = cos(np.pi * x[1])
            c3 = cos(np.pi * x[2])
            pis = np.pi ** 2
            value[0] = -pis * 2 * s1s * ((c2 ** 2 - s2s) * s3s + s2s * (c3 ** 2 - s3s))
            value[1] = pis * 4 * s1 * c1 * s2 * c2 * s3s
            value[2] = pis * 4 * s1 * c1 * s3 * c3 * s2s

        def value_shape(self):
            return (3,)

    class MyExpressionE(UserExpression):
        def eval(self, value, x):
            s1 = sin(np.pi * x[0])
            s2 = sin(np.pi * x[1])
            s3 = sin(np.pi * x[2])
            s1s = s1 ** 2
            s2s = s2 ** 2
            s3s = s3 ** 2
            value[0] = s1s * s2s * s3s
            value[1] = 0
            value[2] = 0

        def value_shape(self):
            return (3,)
            # minit = MyExpression1(degree=1)

    JccE = MyExpressionccE(degree=1)
    JccE = interpolate(JccE, X)
    # dolfin.plot(J)
    # plt.show()
    JE = MyExpressionE(degree=1)
    JE = interpolate(JE, X)
    # mj= project(minit, X)

    #Hj = project(minit, Y)
    #if plotoutput:
    #    dolfin.plot(Hj)
    #    plt.show()

    def dlt(z):
        # BDF1
        return 1.0 - z
        # BDF2
        # return 1.0-z+0.5*(1.0-z)**2

    # Initial Data and Input functions
    #J = project(Expression(['0.00', '-10.0', '-3.0'], degree=1), X)
    Ej = project(Expression(['0.0', '0.0', '0.0'], degree=1), X)
    Hj = project(Expression(['0.0', '0.0', '0.0'], degree=1), Y)
    #J2 = project(Expression(['0.00', '-10.0', '-3.0'], degree=1), X)

    # Coefficient vector [E,H,phi,psi]
    phiko = np.zeros(nBEM)
    sol = np.concatenate([Ej.vector(), Hj.vector(), phiko])

    # interior operators and mass matrices
    Uli = TrialFunction(X)
    UliT = TestFunction(X)
    M1 = FenicsOperator((inner(Uli, UliT)) * dx)

    Uli = TrialFunction(Y)
    UliT = TestFunction(Y)
    M0 = FenicsOperator((inner(Uli, UliT)) * dx)
    Uli = TrialFunction(X)
    UliT = TestFunction(Y)
    D = FenicsOperator(inner(curl(Uli), (UliT)) * dx)

    # Calderon operator
    calderon = (dlt(0) / tau) ** (-m) * operatorn.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(
        0) / tau)  # calderon operator
    cald = tau ** (-m) * 1.0 / mu * calderon.weak_form()
    B11 = -1 / mu * sqrt(mu / eps) * cald[0, 1];
    B12 = cald[0, 0];
    B21 = - cald[1, 1];
    B22 = mu * np.sqrt(eps / mu) * cald[1, 0];

    # boundary mass matrices
    mass1 = bempp.api.operators.boundary.sparse.identity(brt_space, rwg_space, rbc_space)
    # trace operator
    trace_op = aslinearoperator(trace_matrix)
    G1 = mass1.strong_form() * trace_op
    G1T = trace_op.adjoint() * mass1.strong_form()

    # Definition coupled 3x3 matrix
    blocke1 = np.ndarray([3, 3], dtype=np.object);
    blocke1[0, 0] = (eps / tau + sig) * M1.weak_form() - G1T * B22 * G1;
    blocke1[0, 1] = -1.0 * D.weak_form().transpose();
    blocke1[0, 2] = G1T * B21 - (0.5 / mu) * trace_op.adjoint() * mass1.weak_form().transpose();
    blocke1[1, 0] = 1.0 * D.weak_form();
    blocke1[1, 1] = mu / tau * M0.weak_form();
    blocke1[1, 2] = np.zeros((nFEMH, nBEM))
    blocke1[2, 0] = 0.5 / mu * mass1.weak_form() * trace_op + B12 * G1;
    blocke1[2, 1] = np.zeros((nBEM, nFEMH));
    blocke1[2, 2] = -B11

    Lhs = bempp.api.BlockedDiscreteOperator(np.array(blocke1))

    stop = timeit.default_timer()
    print('Time for initial data and LHS: ', stop - start)
    start = timeit.default_timer()

    # Definition of Convolution Quadrature weights
    storblock = np.ndarray([2, 2], dtype=np.object);  # dummy variable
    wei = np.ndarray([int(L)], dtype=np.object);  # dummy array of B(zeta_l)(zeta_l)**(-m)
    CQweights = np.ndarray([int(N + 1)], dtype=np.object);  # array of the weights CQweights[n]~B_n
    for ell in range(0, int(np.floor(L/2)+1)):  # CF Lubich 1993 On the multistep time discretization of linearinitial-boundary value problemsand their boundary integral equations, Formula (3.10)
        wei[ell]= (dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau) ** (-m) * 1.0 / mu *maxwell.multitrace_operator(trace_space.grid, 1j * sqrt(mu * eps) * dlt(rho * np.exp(2.0 * np.pi * 1j * ell / L)) / tau)#.weak_form()
        #

        print(ell)
        # wei[ell]=((1- rho*np.exp(2*np.pi*1j*ell/L))/h)**(-m)*bempp.api.operators.boundary.sparse.multitrace_identity(trace_space.grid,None,'maxwell').weak_form()
    stop = timeit.default_timer()
    print('Time for Calderon evaluation: ', stop - start)
    start = timeit.default_timer()


    randkoeff = 1j * np.zeros([int(N + 1), 2 * nBEM])  # storage variable for boundary coefficients
    dtmpsiko = 1j * np.zeros([int(N + 1), 2 * nBEM])
    Ekoeff = np.zeros([int(N + 1), nFEME])
    Hkoeff = np.zeros([int(N + 1), nFEMH])
    boundary_rhs = [bempp.api.GridFunction(rwg_space, coefficients=np.ones(nBEM)),
                    bempp.api.GridFunction(bc_space, coefficients=np.ones(nBEM))]
    dtmfunc = [bempp.api.GridFunction(rwg_space, coefficients=np.zeros(nBEM)),
               bempp.api.GridFunction(bc_space, coefficients=np.zeros(nBEM))]
    stop = timeit.default_timer()
    print('Time for Convolution weights: ', stop - start)

    for j in range(0, int(N)):  # time stepping: update from timestep t_j to j+1
        print("------------------Now at time step ", str(j), " ------------------")

        Ejko = sol[:nFEME]  # coefficients of timesstep t_j
        Hjko = sol[nFEME:nFEME + nFEMH]
        Ekoeff[j,:]=Ejko[:]
        Hkoeff[j, :] = Hjko[:]
        randkoeff[j, :nBEM] = sol[nFEME + nFEMH:]  # store the boundary data, store varphi, -gamma_tE wrt to RWG
        randkoeff[j, nBEM:] = - G1.dot(Ejko)
        start = timeit.default_timer()
        # Maxwell update
        dtmpsiko[:, :] = randkoeff[:, :]  # compute \partial_t^m phi,  \partial_t^m psi,
        for k in range(0, m):
            for r in range(0, j + 1):
                dtmpsiko[j + 1 - r, :] = (dtmpsiko[j + 1 - r, :] - dtmpsiko[j + 1 - r - 1, :]) / tau

        boundary_rhs[0].coefficients = np.zeros(nBEM)
        boundary_rhs[1].coefficients = np.zeros(nBEM)

        for kk in range(1, j + 2):  # Convolution, start bei 1, da psiko(0)=0
            dtmfunc[0].coefficients = -dtmpsiko[kk, nBEM:]
            dtmfunc[1].coefficients = np.sqrt(mu * eps) ** (-1) * dtmpsiko[kk, :nBEM]
            boundary_rhs += rho ** (-(j + 1 - kk)) / L * np.real(wei[0] * dtmfunc)
            for ell in range(1, int(np.ceil(L / 2) - 1) + 1):  # it is wei(L-d)=complconj(wei(d))
                boundary_rhs += rho ** (-(j + 1 - kk)) / L * np.real(
                    2 * np.exp(-2.0 * np.pi * 1j * (j + 1 - kk) * ell / L) * wei[ell] * dtmfunc)
            if not (L % 2):
                boundary_rhs += rho ** (-(j + 1 - kk)) / L * np.real((-1) ** (j + 1 - kk) * wei[int(L / 2)] * dtmfunc)
        boundary_rhs[1].coefficients = np.real(np.sqrt(mu * eps) * boundary_rhs[1].coefficients)





        if timeoutput:
            stop = timeit.default_timer()
            print('Time for Convolution', j, ' is ', stop - start)
            start = timeit.default_timer()

            # Right hand side  change of J after half of time
        #if (j < N / 2):
        t = (j + 1) * tau
        Rhs = np.concatenate(
                [eps / tau * M1.weak_form().dot( Ejko) +1.0/3.0*t**3*M1.weak_form().dot(JccE.vector())+t*(sig*t+2.0)*M1.weak_form().dot(JE.vector()) + G1T.dot(1.0 * np.real(boundary_rhs[1].projections(wei[0].dual_to_range_spaces[1]))),
                 mu / tau * M0.weak_form().dot(Hjko), - 1.0 * np.real(boundary_rhs[0].projections(wei[0].dual_to_range_spaces[0])) ])
        #else:
        #    Rhs = np.concatenate(
        #        [eps / tau * M1.weak_form() * Ejko - M1.weak_form() * J2.vector()*tau*j  - G1T.dot(boundary_rhs[nBEM:]),
        #         mu / tau * M0.weak_form() * Hjko, boundary_rhs[:nBEM]])

        nrm_rhs = np.linalg.norm(Rhs)  # norm of r.h.s. vector
        it_count = 0

        def count_iterations(x):
            nonlocal it_count
            it_count += 1

        # if Precond:
        #  sol, info  = gmres(Lhs, Rhs,M=P, tol= tolgmres,callback=count_iterations,x0=sol)
        # else:
        sol, info = gmres(Lhs, Rhs, tol=tolgmres, callback=count_iterations, x0=sol)
        if (info > 0):
            print("Failed to converge after " + str(info) + " iterations")
        else:
            print("Solved system " + str(j) + " in " + str(it_count) + " iterations " + " Size of RHS " + str(nrm_rhs))
        if timeoutput:
            stop = timeit.default_timer()
            print('Time for GMRES ', j, ' is ', stop - start)
            # EH.vector()[:]=sol


    #randkoeff[int(N),:]= np.real(boundary_rhs[:])
    randkoeff[int(N),:nBEM]= np.real(sol[nFEME+nFEMH:])
    randkoeff[int(N), nBEM:2 * nBEM]= - G1.dot(sol[:nFEME])
    Ekoeff[int(N),:]=np.real(sol[:nFEME])
    Hkoeff[int(N),:]=np.real(sol[nFEME:nFEME+nFEMH])
    return (Ekoeff,Hkoeff, randkoeff[:,:nBEM],randkoeff[:,nBEM:2*nBEM]);
    #return(mkoeff,Ekoeff,Hkoeff, boundary_rhs[:nBEM],boundary_rhs[nBEM:2*nBEM])