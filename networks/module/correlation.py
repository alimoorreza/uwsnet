import torch
"""
Performs a batch matrix-matrix product of matrices stored in input and mat2.

input and mat2 must be 3-D tensors each containing the same number of matrices.

If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor, out will be a (b \times n \times p)(b×n×p) tensor.

\text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
out 
i
​
 =input 
i
​
 @mat2 
i
​
 
This operator supports TensorFloat32.

On certain ROCm devices, when using float16 inputs this module will use different precision for backward.

NOTE
"""

class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            # https://pytorch.org/docs/stable/generated/torch.bmm.html
            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)  #
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # If input is a (bsz×3600×ch) tensor, mat2 is a (bsz×ch×3600) tensor, out will be a (bsz×3600×3600) tensor.
            # bmm becomes a correlation between the colum vector of support feature and row vector of query feature
            corr = corr.clamp(min=0)
            corrs.append(corr)

        # corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        # corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        # corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return corrs
