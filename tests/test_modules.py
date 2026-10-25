import torch
import torch.nn.functional as F
import pytest
from model import RCL, DegreePyramid, SheafModule, CayleyOrthogonal


def test_one_input_produces_seven_average_pooled_grades():
    # A ramp makes Avg Pool distinguishable from interpolation or feature blending.
    source = torch.arange(65*33, dtype=torch.float64).reshape(1, 1, 65, 33)
    features = DegreePyramid(6)(source)
    assert len(features) == 7
    assert [x.shape[-2:] for x in features] == [(65,33),(33,17),(17,9),(9,5),(5,3),(3,2),(2,1)]
    torch.testing.assert_close(features[0], source)
    assert features[1][0,0,0,0] == (0+1+33+34)/4
    # The last partial window uses actual samples, without zero-padding dilution.
    assert features[1][0,0,-1,-1] == source[0,0,-1,-1]
    assert features[6][0,0,-1,0] == source[0,0,64,:].mean()
    import pytest
    with pytest.raises(TypeError, match='one'):
        DegreePyramid(6)([source, source])


def test_rcl_fuses_all_degrees_from_one_input():
    layer = RCL(14, 14, 6, output_stride=2).double().eval()
    with torch.no_grad():
        layer.fuse_logits.copy_(torch.arange(1, 8, dtype=torch.float64).log())
    feature = torch.randn(2, 14, 19, 27, dtype=torch.float64, requires_grad=True)
    fused, degrees = layer(feature, return_degrees=True)
    assert fused.shape == (2,14,10,14)
    # Nonuniform weights catch a fixed-mean implementation, even at initialization.
    expected = sum((d+1) * F.interpolate(x, (10,14), mode='bilinear', align_corners=False)
                   for d, x in enumerate(degrees)) / 28
    torch.testing.assert_close(fused, expected)
    grads = torch.autograd.grad(fused.square().mean(), degrees, retain_graph=True)
    assert all(g.abs().sum() > 0 for g in grads)
    fused.square().mean().backward()
    assert feature.grad.abs().sum() > 0
    assert torch.isfinite(layer.fuse_logits.grad).all()
    assert layer.fuse_logits.grad.abs().sum() > 0
    assert layer.fuse_logits.grad.sum().item() == pytest.approx(0, abs=1e-12)


def test_rcl_cauchy_product_linearity_and_kernel_reuse():
    layer = RCL(14, 14, 6).double()
    features = [torch.randn(2, 14, 12-j, 13-j, dtype=torch.float64) for j in range(7)]
    results = layer.graded_action(features)
    assert torch.equal(layer.masks.sum(0), torch.ones_like(layer.masks[0]))
    for d, actual in enumerate(results):
        expected = sum(F.interpolate(F.conv2d(features[d-ell], layer._masked_kernel(ell), padding=1),
                                     size=features[d].shape[-2:], mode='bilinear', align_corners=False)
                       for ell in range(d+1))
        torch.testing.assert_close(actual, expected)
    doubled = layer.graded_action([2*x for x in features])
    for a, b in zip(doubled, results):
        torch.testing.assert_close(a, 2*b)
    results[-1].square().mean().backward()
    for ell in range(7):
        selected = layer.W_RCL.grad * layer._masked_kernel(ell).ne(0)
        assert selected.abs().sum() > 0


def test_cayley_orthogonality():
    q = CayleyOrthogonal(4).double()(torch.randn(2, 7, 4, 4, dtype=torch.float64))
    torch.testing.assert_close(q.transpose(-1, -2) @ q, torch.eye(4, dtype=q.dtype).expand_as(q))


def test_sheaf_energy_laplacian_and_final_restrictions():
    module = SheafModule(5, 3, grid_hw=(3, 4)).double()
    feature = torch.randn(2, 5, 6, 8, dtype=torch.float64, requires_grad=True)
    _, energy, diagnostic = module(feature, return_diagnostics=True)
    h = diagnostic['cochain']
    fv, fu = module._restriction_maps(h)
    torch.testing.assert_close(fv, diagnostic['restriction_v'])
    torch.testing.assert_close(fu, diagnostic['restriction_u'])
    l_h = module._apply_laplacian(h, fv, fu)
    torch.testing.assert_close(energy, .5*(h*l_h).sum((1, 2)))
    assert not torch.allclose(energy, diagnostic['graph_energy'])
    (energy.mean()/module.N).backward()
    assert feature.grad.abs().sum() > 0
    assert module.Phi[0].weight.grad.abs().sum() > 0
    # The constant-restriction special case matches the ordinary graph exactly.
    ident = torch.eye(3, dtype=h.dtype).expand_as(fv)
    torch.testing.assert_close(module.sheaf_energy(h, ident, ident), diagnostic['graph_energy'])


def test_sheaf_comparison_bound_on_same_cochain():
    module = SheafModule(3, 3, grid_hw=(2, 3)).double()
    module._set_patch_cover(6, 9)
    h = torch.randn(2, 6, 3, dtype=torch.float64)
    fv, fu = module._restriction_maps(h)
    u, v = module.edge_index
    s = 2 * module.sheaf_energy(h, fv, fu)
    g = (h[:, v]-h[:, u]).square().sum((1, 2))
    eta2 = (torch.linalg.matrix_norm(fv-fu, ord=2).square() *
            torch.minimum(h[:, v].square().sum(-1), h[:, u].square().sum(-1))).sum(-1)
    assert torch.all((s.sqrt()-g.sqrt()).abs() <= eta2.sqrt() + 1e-9)


def test_sheaf_cover_uses_actual_patch_intersections_and_local_pooling():
    module = SheafModule(1, 1, grid_hw=(3, 4)).double()
    with torch.no_grad():
        module.lift.weight.fill_(1)
        module.lift.bias.zero_()
    source = torch.arange(48, dtype=torch.float64).reshape(1, 1, 6, 8).requires_grad_()
    h = module.patch_cochain(source)
    assert (module.gh, module.gw, module.N) == (3, 4, 12)
    # An independent set-of-pixels oracle tests the nerve, including diagonals.
    patches = [{(y,x) for y in range(y0,y1) for x in range(x0,x1)}
               for y0,y1,x0,x1 in module.patch_bounds.tolist()]
    assert set.union(*patches) == {(y,x) for y in range(6) for x in range(8)}
    for i, patch in enumerate(patches):
        expected = torch.stack([source[0,0,y,x] for y,x in sorted(patch)]).mean()
        torch.testing.assert_close(h[0,i,0], expected)
        for j, other in enumerate(patches):
            assert module.A[i,j].item() == (i != j and bool(patch & other))
    assert module.A[0,5]  # Diagonal windows overlap; a 4-neighbor graph omits this.
    assert not module.A[0,2]  # Boundaries touch, but no actual pixel is shared.
    assert module.edge_index.shape[1] == 29
    torch.testing.assert_close(h[0,0,0], source[0,0,:3,:3].mean())
    assert h[0,0,0] != F.adaptive_avg_pool2d(source, (3,4))[0,0,0,0]
    gradient, = torch.autograd.grad(h[0,0,0], source)
    expected_gradient = torch.zeros_like(source)
    expected_gradient[0,0,:3,:3] = 1/9
    torch.testing.assert_close(gradient, expected_gradient)


def test_sheaf_rebuilds_cover_and_preserves_both_forward_graphs():
    module = SheafModule(2, 3, grid_hw=(3, 4)).double()
    first = torch.randn(2,2,6,8,dtype=torch.float64,requires_grad=True)
    second = torch.randn(2,2,3,4,dtype=torch.float64,requires_grad=True)
    _, energy1, diag1 = module(first, return_diagnostics=True)
    bounds1, edges1 = diag1['patch_bounds'].clone(), diag1['patch_edges'].clone()
    _, energy2, diag2 = module(second, return_diagnostics=True)
    assert not torch.equal(bounds1, diag2['patch_bounds'])
    assert diag2['patch_edges'].shape[1] > edges1.shape[1]
    torch.testing.assert_close(diag1['patch_bounds'], bounds1)
    torch.testing.assert_close(diag1['patch_edges'], edges1)
    (energy1.sum()+energy2.sum()).backward()
    assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0


def test_sheaf_relu_diffusion_and_single_patch_energy():
    module = SheafModule(2, 2).double()
    with torch.no_grad():
        module.lift.weight.zero_()
        module.lift.bias.copy_(torch.tensor([-2.,3.]))
        for mixing in module.Wk:
            mixing.weight.zero_()
    refined, energy, diagnostic = module(torch.zeros(1,2,1,1,dtype=torch.float64),
                                          return_diagnostics=True)
    torch.testing.assert_close(refined.flatten(), torch.tensor([0.,3.],dtype=torch.float64))
    assert module.N == 1 and module.edge_index.shape == (2,0)
    assert energy.item() == 0 and diagnostic['graph_energy'].item() == 0
    assert not any('edge_index' in k or 'patch_bounds' in k or k == 'A'
                   for k in module.state_dict())
    (refined.sum()+energy.sum()).backward()
    assert torch.isfinite(module.lift.bias.grad).all()
