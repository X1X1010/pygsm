import pytest

from pygsm.lot.base import Lot
from pygsm.optimizers import LBFGS, EigenvectorFollow
from pygsm.pes import PES, AvgPES, PenaltyPES


@pytest.mark.parametrize('surface_type', [AvgPES, PenaltyPES])
def test_composite_surface_clones_only_its_constructor_parameters(surface_type):
    lot = Lot(geom=[('H', 0.0, 0.0, 0.0)], states=[(1, 0), (1, 1)])
    surface = surface_type(PES(lot), PES(lot, ad_idx=1), lot)
    if surface_type is PenaltyPES:
        surface.sigma = 3.5
        surface.alpha = 0.8
    new_lot = lot.clone(node_id=4)

    clone = surface.clone(lot=new_lot)
    assert type(clone) is surface_type
    assert clone.pes1.lot is clone.pes2.lot is new_lot
    assert (clone.pes1.ad_idx, clone.pes2.ad_idx) == (0, 1)
    assert surface.lot is lot
    with pytest.raises(TypeError, match='ad_idx'):
        surface.clone(ad_idx=2)

    copied = surface_type.create_pes_from(surface, node_id=5)
    assert copied.pes1.lot is copied.pes2.lot is copied.lot
    assert copied.lot is not lot
    assert copied.lot.node_id == 5
    if surface_type is PenaltyPES:
        assert (clone.sigma, clone.alpha) == (copied.sigma, copied.alpha) == (3.5, 0.8)


@pytest.mark.parametrize('optimizer_type', [EigenvectorFollow, LBFGS])
def test_clone_preserves_forwarded_optimizer_parameters(optimizer_type):
    optimizer = optimizer_type(dmax=0.2, conv_de=0.03, linesearch='backtrack')
    clone = optimizer.clone(dmax=0.1)
    assert clone.dmax == 0.1
    assert optimizer.dmax == 0.2
    assert clone.conv_de == 0.03
    assert clone.linesearch == 'backtrack'
