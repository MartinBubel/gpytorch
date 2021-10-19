import torch


class LaplacianCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, dist_func):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("LaplacianCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("LaplacianCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        unitless_dist = dist_func(x1_, x2_)
        # clone because inplace operations will mess with what's saved for backward
        unitless_dist_ = unitless_dist.clone() if needs_grad else unitless_dist
        covar_mat = unitless_dist_.div_(-1.0).exp_()
        if needs_grad:
            d_output_d_input = unitless_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None
