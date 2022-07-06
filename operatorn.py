"""Definition of the Maxwell boundary operators."""

# pylint: disable-msg=too-many-arguments


def _electric_field_impl(
        domain, range_, dual_to_range, wave_number,
        label, symmetry, parameters, assemble_only_singular_part):
    """ Return the actual electric field operator. """
    from bempp.core.operators.boundary.maxwell import electric_field_ext
    from bempp.api.assembly import ElementaryBoundaryOperator
    from bempp.api.assembly.abstract_boundary_operator import \
        ElementaryAbstractIntegralOperator

    #pylint: disable=protected-access
    return ElementaryBoundaryOperator(
        ElementaryAbstractIntegralOperator(
            electric_field_ext(parameters, domain._impl, range_._impl,
                               dual_to_range._impl,
                               wave_number, "", symmetry),
            domain, range_, dual_to_range),
        parameters=parameters, label=label,
        assemble_only_singular_part=assemble_only_singular_part)

def _magnetic_field_impl(
        domain, range_, dual_to_range, wave_number,
        label, symmetry, parameters, assemble_only_singular_part):
    """ Return the actual magnetic field operator. """
    from bempp.core.operators.boundary.maxwell import magnetic_field_ext
    from bempp.api.assembly import ElementaryBoundaryOperator
    from bempp.api.assembly.abstract_boundary_operator import \
        ElementaryAbstractIntegralOperator

    #pylint: disable=protected-access
    return ElementaryBoundaryOperator(
        ElementaryAbstractIntegralOperator(
            magnetic_field_ext(parameters, domain._impl, range_._impl,
                               dual_to_range._impl,
                               wave_number, "", symmetry),
            domain, range_, dual_to_range),
        parameters=parameters, label=label,
        assemble_only_singular_part=assemble_only_singular_part)

def electric_field(domain, range_, dual_to_range,
                   wave_number,
                   label="EFIE", symmetry='no_symmetry',
                   parameters=None, use_projection_spaces=True,
                   assemble_only_singular_part=False):
    """Return the Maxwell electric field boundary operator.

    Parameters
    ----------
    domain : bempp.api.space.Space
        Domain space.
    range_ : bempp.api.space.Space
        Range space.
    dual_to_range : bempp.api.space.Space
        Dual space to the range space.
    wave_number : complex
        Wavenumber for the Helmholtz problem.
    label : string
        Label for the operator.
    symmetry : string
        Symmetry mode. Possible values are: 'no_symmetry',
        'symmetric', 'hermitian'.
    parameters : bempp.api.common.ParameterList
        Parameters for the operator. If none given the
        default global parameter object `bempp.api.global_parameters`
        is used.
    use_projection_spaces : bool
        Represent operator by projection from higher dimensional space
        if available. This parameter can speed up fast assembly routines,
        such as H-Matrices or FMM (default true).
    assemble_only_singular_part : bool
        When assembled the operator will only contain components for adjacent or
        overlapping test and trial functions (default false).
    """
    from bempp.api.operators.boundary._common import \
        get_wave_operator_with_space_preprocessing
    from bempp.api.space import rewrite_operator_spaces

    try:
        #pylint: disable=protected-access
        hdiv_dual_to_range = dual_to_range._hdiv_space
    except:
        raise ValueError(
            "The dual space must be a valid Nedelec curl-conforming space.")

    return rewrite_operator_spaces(get_wave_operator_with_space_preprocessing(
        _electric_field_impl, domain, range_, hdiv_dual_to_range,
        wave_number, label, symmetry, parameters,
        use_projection_spaces, assemble_only_singular_part),
                                   domain, range_, dual_to_range)

def calderon_electric_field(grid, wave_number, parameters=None):
    """Return a pair (E^2, E) of the squared EFIE operator E^2 and E itself"""
    import bempp.api

    class EfieSquared(bempp.api.assembly.BoundaryOperator):
        """Implementation of the squared electric field operator."""

        def __init__(self, grid, wave_number, parameters):
            from bempp.api.assembly import InverseSparseDiscreteBoundaryOperator
            from bempp.api.space import project_operator

            bc_space = bempp.api.function_space(grid, "BC", 0)
            rbc_space = bempp.api.function_space(grid, "RBC", 0)
            rwg_space = bempp.api.function_space(grid, "B-RWG", 0)
            snc_space = bempp.api.function_space(grid, "B-SNC", 0)
            rwg_bary_space = bempp.api.function_space(
                grid.barycentric_grid(), "RWG", 0)
            snc_bary_space = bempp.api.function_space(
                grid.barycentric_grid(), "SNC", 0)
            super(EfieSquared, self).__init__(
                rwg_space, rwg_space, rbc_space, label="EFIE_SQUARED")

            self._efie_fine = electric_field(
                rwg_bary_space, rwg_bary_space, snc_bary_space, wave_number,
                parameters=parameters)
            self._efie = project_operator(
                self._efie_fine, domain=rwg_space,
                range_=rwg_space, dual_to_range=snc_space)
            self._efie2 = project_operator(
                self._efie_fine, domain=bc_space,
                range_=rwg_space, dual_to_range=rbc_space)
            self._ident = bempp.api.operators.boundary.sparse.identity(
                bc_space, rwg_space, snc_space)
            self._inv_ident = InverseSparseDiscreteBoundaryOperator(
                self._ident.weak_form())

        def _weak_form_impl(self):

            efie_weak = self._efie.weak_form()
            efie2_weak = self._efie2.weak_form()

            return efie2_weak * self._inv_ident * efie_weak

    operator = EfieSquared(grid, wave_number, parameters)
    #pylint: disable=protected-access
    return operator, operator._efie2


def magnetic_field(domain, range_, dual_to_range,
                   wave_number,
                   label="MFIE", symmetry='no_symmetry',
                   parameters=None,
                   use_projection_spaces=True,
                   assemble_only_singular_part=False):
    """Return the Maxwell magnetic field boundary operator.

    Parameters
    ----------
    domain : bempp.api.space.Space
        Domain space.
    range_ : bempp.api.space.Space
        Range space.
    dual_to_range : bempp.api.space.Space
        Dual space to the range space.
    wave_number : complex
        Wavenumber for the Helmholtz problem.
    label : string
        Label for the operator.
    symmetry : string
        Symmetry mode. Possible values are: 'no_symmetry',
        'symmetric', 'hermitian'.
    parameters : bempp.api.common.ParameterList
        Parameters for the operator. If none given the
        default global parameter object `bempp.api.global_parameters`
        is used.
    use_projection_spaces : bool
        Represent operator by projection from higher dimensional space
        if available. This parameter can speed up fast assembly routines,
        such as H-Matrices or FMM (default true).
    assemble_only_singular_part : bool
        When assembled the operator will only contain components for adjacent or
        overlapping test and trial functions (default false).

    """

    from bempp.api.operators.boundary._common import \
        get_wave_operator_with_space_preprocessing
    from bempp.api.space import rewrite_operator_spaces

    try:
        #pylint: disable=protected-access
        hdiv_dual_to_range = dual_to_range._hdiv_space
    except:
        raise ValueError(
            "The dual space must be a valid Nedelec curl-conforming space.")

    return rewrite_operator_spaces(get_wave_operator_with_space_preprocessing(
        _magnetic_field_impl, domain, range_, hdiv_dual_to_range,
        wave_number, label, symmetry, parameters,
        use_projection_spaces, assemble_only_singular_part),
                                   domain, range_, dual_to_range)

def multitrace_operator(grid, wave_number, 
        parameters=None, target=None):
    """Assemble the multitrace operator for Maxwell."""

    from bempp.api.assembly import BlockedOperator
    from bempp.api.space import project_operator
    import bempp.api

    blocked_operator = BlockedOperator(2, 2)

    rwg_space_fine = bempp.api.function_space(grid.barycentric_grid(), "RWG", 0)
    rwg_space = bempp.api.function_space(grid, "B-RWG", 0)
    bc_space = bempp.api.function_space(grid, "BC", 0)
    rbc_space = bempp.api.function_space(grid, "RBC", 0)
    snc_space = bempp.api.function_space(grid, "B-SNC", 0)
    snc_space_fine = bempp.api.function_space(grid.barycentric_grid(), "SNC", 0)

    if target is None:

        efie_fine = electric_field(
            rwg_space_fine, rwg_space_fine,
            snc_space_fine, wave_number, parameters=parameters)
        mfie_fine = magnetic_field(
            rwg_space_fine, rwg_space_fine,
            snc_space_fine, wave_number, parameters=parameters)

        #blocked_operator[0, 0] = project_operator(
            #mfie_fine,bc_space , rwg_space, rbc_space)
        #blocked_operator[0, 1] = project_operator(
            #efie_fine, rwg_space, rwg_space, rbc_space)
        #blocked_operator[1, 0] = -1 * project_operator(
            #efie_fine, bc_space, bc_space, snc_space)
        #blocked_operator[1, 1] = project_operator(
            #mfie_fine, rwg_space, bc_space, snc_space)
        
        
        blocked_operator[0, 0] = project_operator(
            mfie_fine,rwg_space , rwg_space, rbc_space)
        blocked_operator[0, 1] = project_operator(
            efie_fine, bc_space, rwg_space, rbc_space)
        blocked_operator[1, 0] = -1 * project_operator(
            efie_fine, rwg_space, bc_space, snc_space)
        blocked_operator[1, 1] = project_operator(
            mfie_fine, bc_space, bc_space, snc_space)
    else:

        rwg_space_fine_target = bempp.api.function_space(target.barycentric_grid(), "RWG", 0)
        rwg_space_target = bempp.api.function_space(target, "B-RWG", 0)
        bc_space_target = bempp.api.function_space(target, "BC", 0)
        rbc_space_target = bempp.api.function_space(target, "RBC", 0)
        snc_space_target = bempp.api.function_space(target, "B-SNC", 0)
        snc_space_fine_target = bempp.api.function_space(target.barycentric_grid(), "SNC", 0)

        efie_fine = electric_field(
            rwg_space_fine, rwg_space_fine_target,
            snc_space_fine_target, wave_number, parameters=parameters)
        mfie_fine = magnetic_field(
            rwg_space_fine, rwg_space_fine_target,
            snc_space_fine_target, wave_number, parameters=parameters)

        blocked_operator[0, 0] = project_operator(
            mfie_fine, rwg_space, rwg_space_target, rbc_space_target)
        blocked_operator[0, 1] = project_operator(
            efie_fine, bc_space, rwg_space_target, rbc_space_target)
        blocked_operator[1, 0] = -1 * project_operator(
            efie_fine, rwg_space, bc_space_target, snc_space_target)
        blocked_operator[1, 1] = project_operator(
            mfie_fine, bc_space, bc_space_target, snc_space_target)

    return blocked_operator