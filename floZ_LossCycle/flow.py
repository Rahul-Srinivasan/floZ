import torch

import logging
logger = logging.getLogger(__name__)

from torch import nn
from torch.nn.init import uniform_ as _init_uniform
from nflows.flows.base import Flow as _Flow
from nflows.transforms import Transform
from nflows.distributions.normal import Distribution, StandardNormal

def _get_transform_from_string(num_dim, flow_type, flow_dict):

    from nflows.nn import nets as nets

    from nflows.transforms.base             import CompositeTransform
    from nflows.transforms.lu               import LULinear
    from nflows.transforms.permutations     import ReversePermutation, RandomPermutation
    from nflows.transforms.normalization    import BatchNorm
    from nflows.transforms.svd              import SVDLinear
    from nflows.transforms.autoregressive   import (MaskedAffineAutoregressiveTransform,
                                                    MaskedPiecewiseLinearAutoregressiveTransform,
                                                    MaskedPiecewiseQuadraticAutoregressiveTransform,
                                                    MaskedPiecewiseCubicAutoregressiveTransform,
                                                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform)
    from nflows.transforms.coupling         import (AdditiveCouplingTransform,
                                                    AffineCouplingTransform,
                                                    PiecewiseLinearCouplingTransform,
                                                    PiecewiseQuadraticCouplingTransform,
                                                    PiecewiseCubicCouplingTransform,
                                                    PiecewiseRationalQuadraticCouplingTransform)

    # get settings
    num_layers              = flow_dict.get('num_layers',               8)
    hidden_features         = flow_dict.get('hidden_features',          32)
    num_blocks              = flow_dict.get('num_blocks',               2)
    num_bins                = flow_dict.get('num_bins',                 25)
    num_householder         = flow_dict.get('num_householder',          10)
    tail_bound              = flow_dict.get('tail_bound',               5)
    random_mask             = flow_dict.get('random_mask',              False)
    context_features        = flow_dict.get('context_features',         None)
    use_residual_blocks     = flow_dict.get('use_residual_blocks',      True)
    use_batch_norm          = flow_dict.get('use_batch_norm',           False)
    random_permutation      = flow_dict.get('random_permutation',       False)
    svd_linear              = flow_dict.get('svd_linear',               False)
    preserve_volume         = flow_dict.get('preserve_volume',          False)
    unconditional_transform = flow_dict.get('unconditional_transform',  False)
    spline_kind             = flow_dict.get('spline_kind',              'quadratic')

    # check flow_type
    if flow_type not in ['maf', 'nvp', 'nsf']:
        logger.error(f"ERROR: Unknown flow type {flow_type}. Please use one of the followings: 'maf', 'nvp', 'nsf'.")
        raise ValueError(f"Unknown flow type {flow_type}. Please use one of the followings: 'maf', 'nvp', 'nsf'.")

    #ResNet function (might be needed)
    def create_resnet(in_features, out_features):
        return nets.ResidualNet(in_features, out_features,
                                hidden_features=hidden_features,
                                num_blocks=num_blocks,
                                use_batch_norm=use_batch_norm)

    # initialize list of transformations
    transforms = []

    ### MAF
    if flow_type == 'maf':

        for _ in range(num_layers):

            # permutation
            if random_permutation:
                transforms.append(RandomPermutation(features=num_dim))
            else:
                transforms.append(ReversePermutation(features=num_dim))

#            # linear tranform
#            if svd_linear:
#                transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
#            else:
#                transforms.append(LULinear(features=num_dim, identity_init=True))

            # maf
            if spline_kind == 'affine':
                transforms.append(MaskedAffineAutoregressiveTransform(features=num_dim,
                                                                      hidden_features=hidden_features,
                                                                      num_blocks=num_blocks,
                                                                      context_features=context_features,
                                                                      use_residual_blocks=use_residual_blocks,
                                                                      random_mask=random_mask,
                                                                      use_batch_norm=use_batch_norm))
            elif spline_kind == 'linear':
                transforms.append(MaskedPiecewiseLinearAutoregressiveTransform(features=num_dim,
                                                                               hidden_features=hidden_features,
                                                                               num_blocks=num_blocks,
                                                                               context_features=context_features,
                                                                               use_residual_blocks=use_residual_blocks,
                                                                               random_mask=random_mask,
                                                                               use_batch_norm=use_batch_norm))
            elif spline_kind == 'quadratic':
                transforms.append(MaskedPiecewiseQuadraticAutoregressiveTransform(features=num_dim,
                                                                                  hidden_features=hidden_features,
                                                                                  num_blocks=num_blocks,
                                                                                  num_bins=num_bins,
                                                                                  tails='linear',
                                                                                  tail_bound=tail_bound,
                                                                                  context_features=context_features,
                                                                                  use_residual_blocks=use_residual_blocks,
                                                                                  random_mask=random_mask,
                                                                                  use_batch_norm=use_batch_norm))
            elif spline_kind == 'cubic':
                transforms.append(MaskedPiecewiseCubicAutoregressiveTransform(features=num_dim,
                                                                              hidden_features=hidden_features,
                                                                              num_blocks=num_blocks,
                                                                              context_features=context_features,
                                                                              use_residual_blocks=use_residual_blocks,
                                                                              random_mask=random_mask,
                                                                              use_batch_norm=use_batch_norm))
            elif spline_kind == 'rational':
                transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_dim,
                                                                                          hidden_features=hidden_features,
                                                                                          num_blocks=num_blocks,
                                                                                          num_bins=num_bins,
                                                                                          tails='linear',
                                                                                          tail_bound=tail_bound,
                                                                                          context_features=context_features,
                                                                                          use_residual_blocks=use_residual_blocks,
                                                                                          random_mask=random_mask,
                                                                                          use_batch_norm=use_batch_norm))
            else:
                logger.error(f"ERROR: Unknown spline type {flow_type}. Please use one of the followings: 'affine', 'linear', 'quadratic', 'cubic', 'rational'.")
                raise ValueError(f"Unknown spline type {flow_type}. Please use one of the followings: 'affine', 'linear', 'quadratic', 'cubic', 'rational'.")

            # batch normalization
            if use_batch_norm:
                transforms.append(BatchNorm(features=num_dim))

        # permutation
        if random_permutation:
            transforms.append(RandomPermutation(features=num_dim))
        else:
            transforms.append(ReversePermutation(features=num_dim))

        # linear tranform
        if svd_linear:
            transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
        else:
            transforms.append(LULinear(features=num_dim, identity_init=True))

    ### NVP
    elif flow_type == 'nvp':

        mask = torch.ones(num_dim)
        mask[::2] = -1

        for _ in range(num_layers):

            # permutation
            if random_permutation:
                transforms.append(RandomPermutation(features=num_dim))
            else:
                transforms.append(ReversePermutation(features=num_dim))

            # linear tranform
            if svd_linear:
                transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
            else:
                transforms.append(LULinear(features=num_dim, identity_init=True))

            # nvp
            if preserve_volume:
                transforms.append(AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_resnet))
            else:
                transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=create_resnet))
            mask *= -1

            # batch normalization
            if use_batch_norm:
                transforms.append(BatchNorm(features=num_dim))

        # permutation
        if random_permutation:
            transforms.append(RandomPermutation(features=num_dim))
        else:
            transforms.append(ReversePermutation(features=num_dim))

        # linear tranform
        if svd_linear:
            transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
        else:
            transforms.append(LULinear(features=num_dim, identity_init=True))

    ### NSF
    elif flow_type == 'nsf':


        mask = torch.ones(num_dim)
        mask[::2] = -1

        for _ in range(num_layers):

            # permutation
            if random_permutation:
                transforms.append(RandomPermutation(features=num_dim))
            else:
                transforms.append(ReversePermutation(features=num_dim))

            # linear tranform
            if svd_linear:
                transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
            else:
                transforms.append(LULinear(features=num_dim, identity_init=True))

            # nsf
            if spline_kind == 'linear':
                transforms.append(PiecewiseLinearCouplingTransform(mask,
                                                                   transform_net_create_fn=create_resnet,
                                                                   num_bins=num_bins,
                                                                   tails='linear',
                                                                   tail_bound=tail_bound,
                                                                   apply_unconditional_transform=unconditional_transform))
            elif spline_kind == 'quadratic':
                transforms.append(PiecewiseQuadraticCouplingTransform(mask,
                                                                      transform_net_create_fn=create_resnet,
                                                                      num_bins=num_bins,
                                                                      tails='linear',
                                                                      tail_bound=tail_bound,
                                                                      apply_unconditional_transform=unconditional_transform))
            elif spline_kind == 'cubic':
                transforms.append(PiecewiseCubicCouplingTransform(mask,
                                                                  transform_net_create_fn=create_resnet,
                                                                  num_bins=num_bins,
                                                                  tails='linear',
                                                                  tail_bound=tail_bound,
                                                                  apply_unconditional_transform=unconditional_transform))
            elif spline_kind == 'rational':
                transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask,
                                                                              transform_net_create_fn=create_resnet,
                                                                              num_bins=num_bins,
                                                                              tails='linear',
                                                                              tail_bound=tail_bound,
                                                                              apply_unconditional_transform=unconditional_transform))
            else:
                logger.error(f"ERROR: Unknown spline type {flow_type}. Please use one of the followings: 'linear', 'quadratic', 'cubic', 'rational'.")
                raise ValueError(f"Unknown spline type {flow_type}. Please use one of the followings: 'linear', 'quadratic', 'cubic', 'rational'.")

            mask *= -1

            # batch normalization
            if use_batch_norm:
                transforms.append(BatchNorm(features=num_dim))

        # permutation
        if random_permutation:
            transforms.append(RandomPermutation(features=num_dim))
        else:
            transforms.append(ReversePermutation(features=num_dim))

        # linear tranform
        if svd_linear:
            transforms.append(SVDLinear(features=num_dim, num_householder=num_householder, identity_init=True))
        else:
            transforms.append(LULinear(features=num_dim, identity_init=True))

    return CompositeTransform(transforms)


class Flow(_Flow):

    def __init__(self,
                 ndim,
                 base_dist          = 'normal',
                 transform          = 'maf',
                 transform_dict     = {},
                 device             = None,
                 **kwargs):
                     
        # get tranformation
        if isinstance(transform, Transform):
            transform   = transform
        elif isinstance(transform, str):
            transform   = _get_transform_from_string(ndim, transform, transform_dict)
        else:
            logger.info("ERROR: Unable to initialize transformation for normalizing flow. Flow argument must be a string or a nflows.transforms.Transform object.")
            raise ValueError("Unable to initialize transformation for normalizing flow. Flow argument must be a string or a nflows.transforms.Transform object.")

        # get base distribution
        if isinstance(base_dist, Distribution):
            base_dist   = base_dist
        elif isinstance(base_dist, str):
            if base_dist == 'normal':
                base_dist = StandardNormal(shape=[ndim])
            else:
                logger.info("ERROR: Unable to initialize base distribution. Flow argument must be a string ('normal') or a nflows.distributions.Distribution object.")
                raise ValueError("Unable to initialize base distribution. Flow argument must be a string ('normal') or a nflows.distributions.Distribution object.")
        else:
            logger.info("ERROR: Unable to initialize base distribution. Flow argument must be a string ('normal') or a nflows.distributions.Distribution object.")
            raise ValueError("Unable to initialize base distribution. Flow argument must be a string ('normal') or a nflows.distributions.Distribution object.")

        # call __init__ of nflows.Flow
        super(Flow, self).__init__(transform=transform, distribution=base_dist)