from __future__ import division
from dolfin import *
import numpy as np
import bempp.api
from fenics import *
#from fenics_rhs import FenicsRHS
import matplotlib.pyplot as plt
from scipy.special import comb 
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import gmres
from bempp.api.fenics_interface import FenicsOperator
from bempp.api import as_matrix
from haha import *
from bempp.api.operators.boundary import maxwell
import operatorn
import timeit

if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

start = timeit.default_timer()
bempp.api.global_parameters.assembly.boundary_operator_assembly_type="dense"

timeoutput=True #  time measurements 
plotoutput=False
saveoutput=False
Precond= False
h=2    #space discretization
N=3.0 #time discretization

tolgmres=10**(-4) #tolerance gmres

T=0.5  #material parameters
eps=1.0
mu=1.0
sig=0.1
alpha=1.0
Ce=1.0
theta=1.0


tau=float(T)/N
meshh = UnitCubeMesh(h,h,h)
 
m=2      #CQ approximation parameters
rho=tolgmres**(0.5/N)
L=2*N

#spaces
Xr = FiniteElement( "N1curl",meshh.ufl_cell(),  1 )
X= FunctionSpace(meshh, Xr)
fenics_space=X
Yr =VectorElement( "DG",meshh.ufl_cell(), 0 )
Y= FunctionSpace(meshh, Yr)

trace_space, trace_matrix = n1curl_to_rt0_tangential_trace(fenics_space)  #trace space and restriction matrix

bc_space=bempp.api.function_space(trace_space.grid,"BC",0) #domain spaces
rwg_space=bempp.api.function_space(trace_space.grid,"B-RWG",0)
snc_space=bempp.api.function_space(trace_space.grid,"B-SNC",0) # dual to range spaces
rbc_space=bempp.api.function_space(trace_space.grid,"RBC",0)
brt_space=bempp.api.function_space(trace_space.grid,"B-RT",0)

nBEM = trace_space.global_dof_count #DOFs
nFEME  =fenics_space.dim()
nFEMH = Y.dim()
class MyExpression0(Expression):   # Initial value function
      def eval(self, value, x):
            sqnx = (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)+(x[2]-0.5)*(x[2]-0.5)
            r=0.3
            B = 0;  
            if sqnx<=r**2:
                A = 20.0*(r**2-sqnx)**2/r**4;
                value[0] = 0.0#*(x[0]-0.5)/(A*A + sqnx);
                value[1] = 0.0#*(x[1]-0.5)/(A*A + sqnx);
                value[2] = -A#*(x[2]-0.5)/(A*A + sqnx);
            else:
                value[0] = 0.0
                value[1] = 0.0
                value[2] = 0.0
      def value_shape(self):
            return (3,)
      
#Expression(['(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)+(x[2]-0.5)*(x[2]-0.5)','0.0','0.0'],degree=1)
minit = MyExpression0(degree=1)
#Ej= project(minit,X)
#J= project(minit,X)
Hj= project(minit,Y)
if plotoutput:  
  dolfin.plot(Hj)
  plt.show()


def dlt(z):
    # BDF1
    return 1.0-z
    # BDF2
    #return 1.0-z+0.5*(1.0-z)**2
 
# Initial Data and Input functions 
J=  project(Expression(['0.00','1.0','0.0'],degree=1),X) 
Ej= project(Expression(['0.0','0.0','0.0'],degree=1),X) 
Hj= project(Expression(['1.0','0.0','0.0'],degree=1),Y)
J2=  project(Expression(['0.00','1.0','00.0'],degree=1),X) 


# Coefficient vector [E,H,phi,psi]
phiko=np.zeros(nBEM)
sol=np.concatenate([Ej.vector(),Hj.vector(),phiko])


#interior operators and mass matrices
Uli = TrialFunction(fenics_space)
UliT = TestFunction(fenics_space)
M1= FenicsOperator( ( inner(Uli,UliT)) *  dx  )

Uli = TrialFunction(Y)
UliT = TestFunction(Y)
M0= FenicsOperator( ( inner(Uli,UliT)) *  dx  )
Uli = TrialFunction(fenics_space)
UliT = TestFunction(Y)
D= FenicsOperator( inner( curl(Uli), (UliT))*dx)

#Calderon operator
calderon = ( dlt(0)/tau)**(-m) *operatorn.multitrace_operator(trace_space.grid, 1j* sqrt(mu*eps)*dlt(0)/tau) # calderon operator 
cald= tau**(-m)*1.0/ mu * calderon.weak_form()
B11=  -1/mu* sqrt(mu/eps)*cald[0,1]   ;                                     B12= cald[0,0] ;
B21=    - cald[1,1]   ;                                    B22=  mu*np.sqrt(eps/mu)*cald[1,0] ;


# boundary mass matrices
mass1 = bempp.api.operators.boundary.sparse.identity(brt_space ,rwg_space, rbc_space)
# trace operator
trace_op = aslinearoperator(trace_matrix)
G1= mass1.strong_form()*trace_op
G1T= trace_op.adjoint()*mass1.strong_form()

# Definition coupled 3x3 matrix
blocke1=np.ndarray([3,3],dtype=np.object);
blocke1[0,0] = (eps/tau+sig)*M1.weak_form()-G1T* B22*G1;     	blocke1[0,1] = -1.0*D.weak_form().transpose();  		blocke1[0,2] =  G1T*B21-(0.5/mu)*trace_op.adjoint()*mass1.weak_form().transpose();
blocke1[1, 0] = 1.0 * D.weak_form();                                    blocke1[1,1] = mu/tau*M0.weak_form();                   blocke1[1,2] =  np.zeros((nFEMH,nBEM))
blocke1[2,0] = 0.5/mu* mass1.weak_form() * trace_op+ B12*G1; 		            blocke1[2,1] = np.zeros((nBEM,nFEMH));	              blocke1[2,2] = -B11


Lhs = bempp.api.BlockedDiscreteOperator(np.array(blocke1))

stop = timeit.default_timer()
print('Time for initial data and LHS: ', stop - start)  
start = timeit.default_timer()

# Definition of Convolution Quadrature weights 
storblock=np.ndarray([2,2],dtype=np.object); # dummy variable 
wei=np.ndarray([int(L)],dtype=np.object); # dummy array of B(zeta_l)(zeta_l)**(-m)
CQweights=np.ndarray([int(N+1)],dtype=np.object); # array of the weights CQweights[n]~B_n

for ell in range(0,int(L)): # CF Lubich 1993 On the multistep time discretization of linearinitial-boundary value problemsand their boundary integral equations, Formula (3.10)
  calderon=maxwell.multitrace_operator(trace_space.grid,1j*sqrt(mu*eps)* dlt( rho*np.exp(2.0*np.pi*1j*ell/L))/tau)
  cald = (dlt(rho*np.exp(2.0*np.pi*1j*ell/L))/tau)**(-m)*1.0/mu*calderon.weak_form()
  storblock[0, 0] = -1.0 / mu * np.sqrt(mu / eps) * cald[0, 1]
  storblock[0, 1] = cald[0, 0]
  storblock[1, 0] = -cald[1, 1]
  storblock[1, 1] = mu * np.sqrt(eps / mu) * cald[1, 0]
  wei[ell]= bempp.api.as_matrix( bempp.api.BlockedDiscreteOperator(np.array(storblock)))
  #wei[ell]=((1- rho*np.exp(2*np.pi*1j*ell/L))/h)**(-m)*bempp.api.operators.boundary.sparse.multitrace_identity(trace_space.grid,None,'maxwell').weak_form()
stop = timeit.default_timer()
print('Time for Calderon evaluation: ', stop - start)  
start = timeit.default_timer()
for n in range(0,int(N+1)): 
  CQweights[n]=wei[0] # Fourier Transform 
  for ell in range(1,int(L)): 
    CQweights[n]=CQweights[n]+wei[ell]*np.exp(-2.0*np.pi*1j*n*ell/L)
  CQweights[n]= rho**(-n)/L* CQweights[n] 

randkoeff=1j*np.zeros([int(N+1),2*nBEM]) # storage variable for boundary coefficients
dtmpsiko=1j*np.zeros([int(N+1),2*nBEM])


## Although it is not a boundary operator we can use
## the SparseInverseDiscreteBoundaryOperator function from
## BEM++ to turn its LU decomposition into a linear operator.
#P1 = bempp.api.InverseSparseDiscreteBoundaryOperator(blocke1[0,0].sparse_operator.tocsc())
#
## For the Calderon operator we use a simple mass matrix preconditioner.
## This is sufficient for smaller low-frequency problems.
## Or we use a Calderon Preconditioner
#P2=blocke1[3,3]#bempp.api.operators.boundary.sparse.identity(rwg_space, bc_space, snc_space).weak_form()) # or Identity matrix
#P3=bempp.api.InverseSparseDiscreteBoundaryOperator(bempp.api.operators.boundary.sparse.identity(rwg_space, bc_space, rbc_space).weak_form())
#P4=bempp.api.InverseSparseDiscreteBoundaryOperator(bempp.api.operators.boundary.sparse.identity(bc_space, rwg_space, snc_space).weak_form())
#from scipy.sparse.linalg import LinearOperator
## Create a block diagonal preconditioner object using the Scipy LinearOperator class
#def apply_prec(x):
#    """Apply the block diagonal preconditioner"""
#    nfem = P1.shape[1]
#    nbem = P3.shape[1]
#
#    res1 = (eps/tau+sig)**(-1)*P1.dot(x[:nfem])
#    res2 = (mu/tau)**(-1)*P1.dot(x[nfem:2*nfem])
#    res3 = mu*P3.dot(x[2*nfem:2*nfem+nbem])
#    res4 = P4.dot(x[2*nfem+nbem:])
#    res5 = ( dlt(0)/tau)**(m)*calderon.strong_form().dot(np.concatenate([res3,res4]))
#    res3= 4*mu*dlt(0)**m * res5[nBEM:]
#    res4= -4*dlt(0)**m * res5[:nBEM]
#    return np.concatenate([res1, res2,res3,res4])
#p_shape = (2*nFEM+2*nBEM, 2*nFEM+2*nBEM)
#P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))

stop = timeit.default_timer()
print('Time for Convolution weights: ', stop - start)  

for j in range(0,int(N)):  # time stepping: update from timestep t_j to j+1
  print("------------------Now at time step ",str( j ), " ------------------")

  Ejko= sol[:nFEME] #coefficients of timesstep t_j
  Hjko=sol[nFEME:nFEME+nFEMH]
  randkoeff[j,:nBEM]= sol[nFEME+nFEMH:] # store the boundary data, store varphi, -gamma_tE wrt to RWG
  randkoeff[j,nBEM:]=- G1.dot(Ejko)
  start =timeit.default_timer()
  #Maxwell update
  dtmpsiko[:,:]=randkoeff[:,:] # compute \partial_t^m phi,  \partial_t^m psi,
  for k in range(0,m):
    for r in range(0,j+1):
      dtmpsiko[j+1-r,:]=(dtmpsiko[j+1-r,:]-dtmpsiko[j+1-r-1,:])/tau 
  if timeoutput:  
    stop = timeit.default_timer()
    print('Time for dtmpsiko: ', j ,' is ', stop - start)  
    start = timeit.default_timer()
  boundary_rhs= 1j*np.zeros(2*nBEM)
  for kk in range(0,j+2): # Convolution, start bei 0, da psiko(0)=0 aber gamma_TE vlt nicht...
    boundary_rhs+=CQweights[j+1-kk].dot(dtmpsiko[kk,:])
  if timeoutput :  
    stop = timeit.default_timer()
    print('Time for Convolution', j ,' is ', stop - start)  
    start = timeit.default_timer()  
    
  # Right hand side  change of J after half of time
  if (j<N/2): 
    Rhs=np.concatenate([eps/tau*M1.weak_form()*Ejko-M1.weak_form()*J.vector()-G1T.dot(boundary_rhs[nBEM:]),mu/tau*M0.weak_form()*Hjko,boundary_rhs[:nBEM]])
  else: 
    Rhs=np.concatenate([eps/tau*M1.weak_form()*Ejko-M1.weak_form()*J.vector()-G1T.dot(boundary_rhs[nBEM:]),mu/tau*M0.weak_form()*Hjko,boundary_rhs[:nBEM]])
    
  nrm_rhs = np.linalg.norm(Rhs) # norm of r.h.s. vector
  it_count = 0
  def count_iterations(x):
      global it_count
      it_count += 1
  #if Precond:
  #  sol, info  = gmres(Lhs, Rhs,M=P, tol= tolgmres,callback=count_iterations,x0=sol)
  #else:
  sol, info = gmres(Lhs, Rhs, tol=tolgmres, callback=count_iterations, x0=sol)
  if (info > 0):
      print("Failed to converge after " + str(info) + " iterations")
  else:
      print("Solved system "+str(j) + " in " + str(it_count) + " iterations "+" Size of RHS " + str(nrm_rhs))
  if timeoutput:      
    stop = timeit.default_timer()
    print('Time for GMRES ', j ,' is ', stop - start)  
  #EH.vector()[:]=sol
   

  #plot of E and psi
  if plotoutput: 
      E=dolfin.Function(fenics_space)
      H=dolfin.Function(fenics_space)
      E.vector()[:]=np.ascontiguousarray(sol[:nFEM])
      H.vector()[:]=np.ascontiguousarray(sol[nFEM:2*nFEM])
      #E.vector().set_local(sol[:nFEM])
      neumann_fun = -bempp.api.GridFunction(rwg_space, coefficients=sol[2*nFEM+nBEM:])  #gamma_TE ~ -psi
      #neumann_fun.plot()
      dolfin.plot(mj)
      plt.show()
      dolfin.plot(H)
      plt.show()

#Store solutions
if saveoutput: 
  fidm = File('plots/solution2.pvd')
  E.rename("E", "E")
  #Hrhs.rename("Hrhs", "Hrhs")
  fidm << E, T
  #fidh << Hrhs, t
  #vtkfile = File('plots/solution.xml')
  #vtkfile << E 

