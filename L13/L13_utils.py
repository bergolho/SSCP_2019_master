from fenics import *


def darcy_reduced(mesh):
    """Computes pressure p using Darcy's law in its reduced form

    div(-K *grad(p)) = 0

    and returns velocity u by postprocessing the resulting pressure

    u = -K*grad(p)
    """
    # Define function space
    W = FunctionSpace(mesh, 'P', 1) # pressure space

    # Define permeability and porosity
    kmin = 0.01
    K = Expression('max(exp(-pow((x[1]-0.5-0.1*sin(10*x[0]))/0.1, 2)), 0.01)', degree=1)
    phi = Constant(0.2)

    # Define boundary condition
    pD = Expression('(1-x[0]*x[0])', t=0.0, degree=1)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(W, pD, boundary)

    # Define variational problem
    p = TestFunction(W)
    q = TrialFunction(W)
    f = Constant(0.0)
    n = FacetNormal(mesh)
    a = K * inner(grad(p), grad(q))*dx
    L = f*q*dx

    # Compute solution
    dt = 0.1
    t = 0.0
    p = Function(W)

    solve(a == L, p, bc)

    # Compute velocity
    V = VectorFunctionSpace(mesh, 'P', 1)
    u = project(-K*grad(p), V)

    return u
