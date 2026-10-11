import numpy as np

# TODO remove unecessary arguments: nconstraints, xp, ,...


def NoLineSearch(n, x, fx, g, d, step, xp, constraint_step, parameters, molecule, verbose=False):
    """Take the full step ``x + d * step`` (plus the constraint step) without a line search."""
    x = x + d * step + constraint_step  #
    xyz = molecule.coord_obj.newCartesian(molecule.xyz, x - xp, frozen_atoms=molecule.frozen_atoms, verbose=verbose)

    # use these so molecule xyz doesn't change
    print(' evaluate fx in linesearch')
    fx = molecule.PES.get_energy(xyz)
    gx = molecule.PES.get_gradient(xyz, frozen_atoms=molecule.frozen_atoms)
    g = molecule.coord_obj.calcGrad(xyz, gx)

    result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}
    return result


# TODO might be wise to add to backtrack a condition that says if
# the number of iterations was many and the energy increased
# just return the initial point
def backtrack(nconstraints, x, fx, g, d, step, xp, constraint_step, parameters, molecule, verbose=False):
    """Backtracking line search: shrink the step until the Armijo condition holds."""
    print(' In backtrack')

    # n is the non-constrained
    count = 0
    dec = 0.5
    inc = 2.1

    # project out the constraint
    gc = g.copy()
    for c in molecule.constraints.T:
        gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]

    result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}

    # Compute the initial gradient in the search direction.
    dginit = np.dot(gc.T, d)

    # Make sure that s points to a descent direction.
    if dginit > 0:
        print('[ERROR] not descent direction')
        result['status'] = -2
        return result

    # The initial value of the objective function.
    finit = fx

    dgtest = parameters['ftol'] * dginit

    while True:
        x = xp
        x = x + d * step + constraint_step
        xyz = molecule.coord_obj.newCartesian(molecule.xyz, x - xp, frozen_atoms=molecule.frozen_atoms, verbose=verbose)
        # Evaluate the function and gradient values.
        # use these so molecule xyz doesn't change
        fx = molecule.PES.get_energy(xyz)

        gx = molecule.PES.get_gradient(xyz, frozen_atoms=molecule.frozen_atoms)
        g = molecule.coord_obj.calcGrad(xyz, gx)
        width = 1.0

        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]

        count = count + 1

        # check the sufficient decrease condition (Armijo condition).
        if fx > finit + (step * dgtest) and np.all(
            constraint_step == 0
        ):  # + np.dot(g.T,constraint_step): # doesn't work with constraint :(
            print(' [INFO] decreasing step: does not satisfy sufficient decrease condition.')
            width = dec
            print(f' step {step * width:1.5f}')
        else:
            # check the wolfe condition
            # now g is the gradient of f(xk + step * d)
            dg = np.dot(gc.T, d)
            if dg < parameters['wolfe'] * dginit:
                print(' [INFO] dg = {!r} < parameters.wolfe * dginit = {!r}'.format(dg, parameters['wolfe'] * dginit))
                print(' [INFO] not satisfy wolf condition.')
                width = inc
                print(f' increasing step to step {step * width:1.5f}')
            else:
                # check the strong wolfe condition
                if dg > -parameters['wolfe'] * dginit:
                    print(' [INFO] not satisfy strong wolf condition.')
                    width = dec
                    print(f' decreasing step to {step * width:1.5f}')
                else:
                    result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}
                    return result
        if parameters['max_linesearch'] <= count:
            print(' [INFO] the iteration of linesearch is many')
            result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}
            return result

        if step <= parameters['min_step'] and width <= 1.0:
            result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}
            print(' [INFO] the linesearch step is too small')
            return result
        if step >= parameters['max_step'] and width >= 1.0:  # otherwise step will decrease next step
            print(f' [INFO] the linesearch step is too large, returning with step {step}')
            result = {'status': 0, 'fx': fx, 'g': g, 'step': step, 'x': x, 'molecule': molecule}
            return result

        # update the step
        step = step * width

        # make sure step isn't too large
        if step > parameters['max_step']:
            print(step)
            print(' Manually decreasing step size')
            step = parameters['max_step']
